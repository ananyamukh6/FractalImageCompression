
Install OpenCV:
Following this: https://medium.com/@jaskaranvirdi/setting-up-opencv-and-c-development-environment-in-xcode-b6027728003

brew install opencv
brew install pkg-config
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
pkg-config --cflags --libs opencv4  # This should show some prints


g++ $(pkg-config --cflags --libs opencv4) -std=c++11 test.cpp -o test



Using cmake:

mkdir build
cd build
cmake ..
./src/test_exec


For a more detailed report, please see [here](https://sites.google.com/view/ananyamukherjeehome/image-compression/fractal-compression)
