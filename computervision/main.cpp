#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    Mat result1;
    Mat result2;
    Mat srcMat = imread("/Users/joey/Documents/Study/作业项目/大学生职业生涯规划4/工作与生活的平衡.png",0);
    threshold(srcMat,result1,100,255,THRESH_OTSU);
    adaptiveThreshold(srcMat,result2,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,15,10);
    imshow("source",srcMat);
    imshow("threshold",result1);
    imshow("adaptiveThreshold",result2);
    waitKey(0);
    return 0;
}
