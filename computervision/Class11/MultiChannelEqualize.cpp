#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    Mat src = imread("/Users/joey/Desktop/face.jpg");
    Mat dst;
    
    Mat channels[3];
    split(src, channels);
    
    for (int i = 0; i < 3; i++) {
        equalizeHist(channels[i], channels[i]);
    }
    
    merge(channels,3,dst);
    
    imshow("src", src);
    imshow("channel1",channels[0]);
    imshow("channel2",channels[1]);
    imshow("channel3",channels[2]);
    imshow("dst", dst);
    
    waitKey(0);
    return 0;

}
