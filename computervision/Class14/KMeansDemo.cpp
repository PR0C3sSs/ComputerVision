#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void segColor();
int createMaskByKmeans(cv::Mat src, cv::Mat &mask);
VideoCapture capture("/Users/joey/Desktop/【蔡徐坤】绿幕素材.165410330.mp4");


int createMaskByKmeans(cv::Mat src, cv::Mat & mask)
{
    if (    (mask.type() != CV_8UC1)
        ||    (src.size() != mask.size())
        ) {
        return 0;
    }

    int width = src.cols;
    int height = src.rows;

    int pixNum = width * height;
    int clusterCount = 2;
    Mat labels;
    Mat centers;

    //制作kmeans用的数据
    Mat sampleData = src.reshape(3, pixNum);
    Mat km_data;
    sampleData.convertTo(km_data, CV_32F);

    //执行kmeans
    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
    kmeans(km_data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

    //制作mask
    uchar fg[2] = { 0,255 };
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            mask.at<uchar>(row, col) = fg[labels.at<int>(row*width+col)];
        }
    }
    
    return 0;
}

int main()
{
    VideoCapture capture("/Users/joey/Desktop/1.mp4");
    
    double rate = capture.get(CAP_PROP_FPS);
    Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));

    VideoWriter writer("/Users/joey/Desktop/VideoTest.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), rate, size);

    while (true)
    {
        Mat frame;
        capture >> frame;
        Mat mask = Mat::zeros(frame.size(), CV_8UC1);
        createMaskByKmeans(frame,mask);
        imshow("frame",frame);
        imshow("mask",mask);
        writer.write(mask);
        waitKey(1);
    }
    return 0;
}
