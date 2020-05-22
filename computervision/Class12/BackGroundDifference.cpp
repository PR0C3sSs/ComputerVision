#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    VideoCapture capVideo(0);

    int cnt = 0;

    Mat frame;
    Mat bgMat;
    Mat subMat;
    Mat dstMat;

    while (1) {

        capVideo >> frame;
        cvtColor(frame,frame,COLOR_BGR2GRAY);

        if (cnt== 0) {
            frame.copyTo(bgMat);
        }
        else {
            absdiff(frame, bgMat, subMat);
            threshold(subMat, dstMat,120, 255, THRESH_BINARY);
            
            imshow("result",dstMat);
            imshow("frame",frame);
            waitKey(30);
        }
        cnt++;
    }
    return 0;
}
