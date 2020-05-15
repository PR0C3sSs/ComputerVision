#include <stdio.h>
#include <opencv2/opencv.hpp>

#define gamma 1/2.2

using namespace cv;
using namespace std;

int myGammaCorrection(Mat src, Mat &dst);

int main(){
    Mat src = imread("/Users/joey/Desktop/gtest.jpg",0);
    Mat dst;
    myGammaCorrection(src, dst);
    
    imshow("src",src);
    imshow("dst",dst);
    //李竹老师超级帅
    
    waitKey(0);

    return 0;
}


int myGammaCorrection(Mat src, Mat &dst){
    char lut[256];
    for( int i = 0; i < 256; i++ )
    {
        lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
    }
    
    dst= src.clone();
    
    for (int i = 0; i< dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){
            dst.at<uchar>(i,j) = lut[src.at<uchar>(i,j)];
        }
    }
    return 0;
}
