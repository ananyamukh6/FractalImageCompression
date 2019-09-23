#include "opencv4/opencv2/opencv.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/core/types_c.h"
#include <iostream>
#include <cassert>
#include <chrono>
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


//TODO 
pair<float, float> FindContrastAndBrightness2 (Mat dst, Mat src){
    return pair<float, float>{1.0,0.0};
    
    /*
    # Fit the contrast and the brightness
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    #x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
    return x[1], x[0]*/

    Mat src_flat = src.reshape(0,vector<int>{});
    Mat dst_flat = dst.reshape(0,1);
    Mat out_val;
    //cv::solve(src_flat, dst_flat, out_val, DECOMP_SVD);
    cv::solve(src, dst, out_val, DECOMP_SVD);
    return pair<float, float>{out_val.at<float>(0), out_val.at<float>(1)};
}

//Compression for greyscale images
cv::Mat SliceImage(cv::Mat image, int row_start, int row_end, int col_start, int col_end){
    cv::Range row_range(row_start,row_end); 
    cv::Range col_range(col_start,col_end);
    return image(row_range, col_range);
}
using TransformedBlock = tuple<int, int, int, int, cv::Mat>;
using TransformedBlocks = vector<TransformedBlock>;
TransformedBlocks GenerateAllTransformedBlocks(Mat img, int src_size, int dst_size, int step){
    //Parameters
    static const std::list<int> directions{1, -1};
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

using PerBlockCompressInfo = tuple<int, int, int, int, float, float>;
vector<vector<PerBlockCompressInfo>> Compress(Mat img, int src_size, int dst_size, int step){
    TransformedBlocks transformed_blocks = GenerateAllTransformedBlocks(img, src_size, dst_size, step);
    int i_count = img.rows/dst_size;
    int j_count = img.cols/dst_size;
    float contrast, brightness;
    vector<vector<PerBlockCompressInfo>> transformations(i_count);
    for (int rows = 0; rows < i_count; rows++){
        cout << "Row: " << rows << "/" << i_count << "\n";
        transformations[rows].resize(j_count);
        for (int cols = 0; cols<j_count; cols++){
            auto D = SliceImage(img, rows*dst_size,(rows+1)*dst_size,cols*dst_size,(cols+1)*dst_size);
            float min_d = std::numeric_limits<float>::max();
            for (const auto& tb : transformed_blocks) {
                auto S = get<4>(tb);
                std::tie(contrast, brightness) = FindContrastAndBrightness2(D, S);
                auto S_mod = contrast * S + brightness;
                float d = ImgDist(S_mod, D);
                if (d < min_d){
                    min_d = d;
                    transformations[rows][cols] = make_tuple(get<0>(tb), get<0>(tb), get<2>(tb), get<3>(tb), contrast, brightness);
                }
            }
        }
    }
    return transformations;
}




int main(){
    ReadAndDisplayImage ("lena1.png");
    auto img = ReadImage("lena1.png");
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

    auto transformations = Compress(img, 8, 4, 8);
    return 0;
}



