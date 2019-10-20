#include "opencv4/opencv2/opencv.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/core/types_c.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>
using namespace std;
#include <string>
using namespace cv;
using namespace std::chrono;

cv::Mat ReadImage (string);
void DisplayImage(cv::Mat);
void ReadAndDisplayImage(string);
void test();
/*using RotateFunction = std::function<pair<int,int>(pair<int,int>)>;
enum class Angle;
RotateFunction GetRotateMatrix(Angle);*/
cv::Mat RotateImage(cv::Mat, int);
cv::Mat SliceImage(cv::Mat, int, int, int, int);
float ImgDist(Mat A, Mat B);
using PerBlockCompressInfo = tuple<int, int, bool, int, float, float>;
vector<vector<PerBlockCompressInfo>> Compress(Mat, int, int, int, int);
Mat Decompress(vector<vector<PerBlockCompressInfo>>, int, int, int, int);

//Functions to Read And Display image and manipulate channels
void ReadAndDisplayImage(string image){
    DisplayImage(ReadImage(image));
}

void DisplayImage (cv::Mat image){
    //create image window named "My Image"
    cv::namedWindow("My Image");
    //show the image on window
    cv::imshow("My Image", image);
    cout << "Width : " << image.cols << endl;
    cout << "Height: " << image.rows << endl;
    cv::waitKey(1000);
}

// Reads image as single channel grayscale image. Values between 0.0 to 1.0
cv::Mat ReadImage(string image) {
    cv::Mat img;
    try {
         img = cv::imread(image, cv::IMREAD_GRAYSCALE);
    } catch(...) {
        cout << "Failed to load image" << endl;
        exit(1);
    }
    Mat img_float;
    img.convertTo(img_float, CV_32FC1);
    cv::Mat img_float_normalized;
    cv::normalize(img_float, img_float_normalized, 0, 1, cv::NORM_MINMAX);
    return img_float_normalized;
}

void test(){
    auto img = ReadImage("lena1.png");
    assert(img.channels() == 1);
}

//Transformations
cv::Mat RotateImage(cv::Mat img, int angle){
    angle = angle%360;
    assert(angle == 0 || angle == 90 || angle == 180 || angle == 270);
    cv::Point2f src_center(img.cols/2.0F, img.rows/2.0F);
    cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    cv::Mat dst;
    warpAffine(img, dst, rot_mat, img.size());
    return dst;
}

// TODO: what of flipUD?
cv::Mat FlipLR(cv::Mat img) {
    cv::Mat dst;
    cv::flip(img, dst, 1);
    return dst;
}

cv::Mat Reduce(cv::Mat img, int factor) {
    cv::Mat dst;
    cv::resize(img, dst, cv::Size(img.rows/factor, img.cols/factor));
    return dst;
}

// This func only operates in 0-255 (int mode).
// Is it enough? do we need to be in float mode?
cv::Mat ApplyTransformation(cv::Mat img, bool flip, int angle, float contrast=1.0, float brightness=0.0){
    return contrast*RotateImage((flip ? FlipLR(img) : img), angle) + brightness;
}
// End of transformations


pair<float, float> FindContrastAndBrightness2 (Mat dst, Mat src){
    assert (dst.rows == src.rows && dst.cols == src.cols);

    int num_elems = dst.rows * dst.cols;
    Mat src_flat = src.clone().reshape(0,num_elems);
    Mat ones = Mat(num_elems, 1, CV_32FC1, Scalar(1.0));
    cv::hconcat(src_flat, ones, src_flat); // src_flat.size = num_elems x 2

    Mat dst_flat = dst.clone().reshape(0, num_elems);

    Mat src_flat_inv;
    invert(src_flat, src_flat_inv, DECOMP_SVD); // src_flat_inv.size = 2 x num_elems
    Mat out_val = src_flat_inv * dst_flat;
    return pair<float, float>{out_val.at<float>(0), out_val.at<float>(1)};
}

//Compression for greyscale images
cv::Mat SliceImage(cv::Mat image, int row_start, int row_end, int col_start, int col_end){
    cv::Range row_range(row_start,row_end);
    cv::Range col_range(col_start,col_end);
    return image(row_range, col_range);
}
using TransformedBlock = tuple<int, int, bool, int, cv::Mat>;
using TransformedBlocks = vector<TransformedBlock>;
TransformedBlocks GenerateAllTransformedBlocks(Mat img, int src_size, int dst_size, int step){
    //Parameters
    static const std::list<bool> directions{true, false};
    static const std::list<int> angles{0, 90, 180, 270};
    int factor = src_size/dst_size;
    TransformedBlocks transformed_blocks;
    for (int rows=0; rows<(img.rows - src_size)/step + 1; rows++){
        for (int cols=0; cols<(img.cols-src_size)/step + 1; cols++){
            auto sliced_img = SliceImage(img, rows*step,(rows*step)+src_size, cols*step,(cols*step)+src_size);
            Mat reduced_img = Reduce(sliced_img, factor);
            for (const auto& dir : directions){
                for (const auto& ang : angles){
                    transformed_blocks.push_back(make_tuple(rows, cols, dir, ang, ApplyTransformation(reduced_img, dir, ang)));
                }
            }
        }
    }
    return transformed_blocks;
}

// TODO: implement me
float ImgDist(Mat A, Mat B) {
    int img_Arows = A.rows;
    int img_Acols = A.cols;
    int img_Brows = B.rows;
    int img_Bcols = B.cols;
    float diff;
    float dist = 0.0;
    assert (img_Arows == img_Brows && img_Acols == img_Bcols);
    for (int col = 0; col<img_Acols; col++){
        for (int row = 0; row<img_Arows; row++){
            diff = A.at<float>(col,row) - B.at<float>(col,row);
            dist += (diff*diff);
        }
    }
    return dist;
}

tuple<float, float, float> CompressHelper(TransformedBlock tb, Mat D){
  float contrast, brightness;
  auto S = get<4>(tb);
  std::tie(contrast, brightness) = FindContrastAndBrightness2(D, S);
  auto S_mod = contrast * S + brightness;
  return make_tuple(ImgDist(S_mod, D), contrast, brightness);
}

vector<vector<PerBlockCompressInfo>> Compress(Mat img, int src_size, int dst_size, int step, int num_threads=8){
    TransformedBlocks transformed_blocks = GenerateAllTransformedBlocks(img, src_size, dst_size, step);
    int i_count = img.rows/dst_size;
    int j_count = img.cols/dst_size;
    float contrast, brightness, d;
    vector<vector<PerBlockCompressInfo>> transformations(i_count);
    for (int rows = 0; rows < i_count; rows++){
        cout << "Row: " << rows << "/" << i_count << "\n";
        transformations[rows].resize(j_count);
        for (int cols = 0; cols<j_count; cols++){
            auto D = SliceImage(img, rows*dst_size,(rows+1)*dst_size,cols*dst_size,(cols+1)*dst_size);
            float min_d = std::numeric_limits<float>::max();

            if (num_threads == 1){
              for (const auto& tb : transformed_blocks) {
                  auto tpl = CompressHelper(tb, D);
                  tie(d, contrast, brightness) = tpl;
                  if (d < min_d){
                      min_d = d;
                      transformations[rows][cols] = make_tuple(get<0>(tb), get<1>(tb), get<2>(tb), get<3>(tb), contrast, brightness);
                  }
              }
            } else {
              vector<tuple<float, float, float>> all_results_vector(transformed_blocks.size());
              auto worker = [&transformed_blocks, &D, &all_results_vector, &num_threads](int th_id){
                for (int idx = th_id; idx < all_results_vector.size(); idx += num_threads){
                  all_results_vector[idx] = CompressHelper(transformed_blocks[idx], D);
                }
              };
              vector<thread> all_threads(num_threads);
              for (int th_id = 0; th_id < num_threads; th_id++){
                all_threads[th_id] = std::thread(worker, th_id);
              }

              for (int th_id = 0; th_id < num_threads; th_id++){
                all_threads[th_id].join();
              }

              for (int i = 0; i < all_results_vector.size(); i++){
                tie(d, contrast, brightness) = all_results_vector[i];
                auto tb = transformed_blocks[i];
                if (d < min_d){
                  min_d = d;
                  transformations[rows][cols] = make_tuple(get<0>(tb), get<1>(tb), get<2>(tb), get<3>(tb), contrast, brightness);
                }
              }
          }
        }
    }
    return transformations;
}

void CopySubMat(cv::Mat cur_img, cv::Mat D, int row_start, int row_end, int col_start, int col_end) {
  assert ((D.rows == (row_end-row_start)) && (D.cols == (col_end-col_start)));
  Mat subMat = cur_img.colRange(col_start, col_end).rowRange(row_start, row_end);
  D.copyTo(subMat);
  return;
}

Mat Decompress(vector<vector<PerBlockCompressInfo>> transformations, int src_size, int dst_size, int step, int nb_iter=16){
  int factor = src_size / dst_size;
  int dim1 = transformations.size();
  int dim2 = transformations[0].size();
  int height = dim1 * dst_size;
  int width = dim2 * dst_size;
  Mat old_img(height, width, CV_32FC1);
  Mat cur_img(height, width, CV_32FC1);
  randu(old_img, Scalar(0.0), Scalar(1.0));
  int k, l;
  bool flip;
  int angle;
  float contrast, brightness;

  bool something_changed = false;
  for (int i_iter = 0; i_iter < nb_iter; i_iter++){
    cout << "Decoding iter: " << i_iter << "\n";
    for (int i = 0; i < dim1; i++){
      for (int j = 0; j < dim2; j++){
        tie(k, l, flip, angle, contrast, brightness) = transformations[i][j];
        auto S = Reduce(SliceImage(old_img, k*step, k*step+src_size, l*step, l*step+src_size), factor);
        auto D = ApplyTransformation(S, flip, angle, contrast, brightness);
        CopySubMat(cur_img, D, i*dst_size, (i+1)*dst_size, j*dst_size, (j+1)*dst_size);
      }
    }
    DisplayImage(cur_img);
    old_img = cur_img;
  }
  return cur_img;
}

// https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html
double getPSNR(const Mat& I1, const Mat& I2)
{
  // Assuming images in range 0-1 is being input
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /(double)(I1.channels() * I1.total());
        double psnr = 20.0*log10((1)/mse);
        return psnr;
    }
}



int main(){
    ReadAndDisplayImage ("lena2.png");
    auto img = ReadImage("lena2.png");
    //DisplayImage(img);
    cout << "Channels: " << (img.channels()) << "\n";
    //cv::Mat rot_img = RotateImage (img, 90);
    //DisplayImage(rot_img);
    //DisplayImage(FlipLR(img));
    //DisplayImage(Reduce(img, 2));
    //auto out = rotate({{1,2,3}, {4,5,6}, {7,8,9}});
    //DisplayImage(img);
    auto t = ApplyTransformation(img, false, 0, 2.0);
    cout << "ApplyTransformation IMAGE. TYPE: " << t.type() << "\n";
    DisplayImage(t);
    Scalar pixel = img.at<float>(0, 0);
    cout << pixel.val[0] << "----\n";

    pixel = t.at<float>(0, 0);
    cout << pixel.val[0] << "----\n";
    GenerateAllTransformedBlocks(img, 8,4,8);

    auto start = std::chrono::high_resolution_clock::now();
    auto transformations = Compress(img, 8, 4, 8, 8);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout << "Done compressing in: " << elapsed.count() << "\n";

    start = std::chrono::high_resolution_clock::now();
    auto decompressed = Decompress(transformations, 8, 4, 8);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    cout << "Done decompression in: " << elapsed.count() << "\n";

    cout << "PSNR: " << getPSNR(img, decompressed) << "\n";

    DisplayImage(decompressed);

    // TODO move to a function
    Mat ucharImg;
    decompressed.convertTo(ucharImg, CV_8U, 255.0);
    imwrite("FINAL.jpg", ucharImg);
    return 0;
}
