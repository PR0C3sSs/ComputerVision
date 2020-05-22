#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int gaussianThreshold(Mat srcMat, Mat avgMat, Mat varMat, float weight, Mat & dstMat)
{
    int srcI;
    int avgI;
    int rows = srcMat.rows;
    int cols = srcMat.cols;

    for (int h = 0; h < rows; h++)
    {
        for (int w = 0; w < cols; w++)
        {
            srcI = srcMat.at<uchar>(h, w);
            avgI = avgMat.at<uchar>(h, w);
            int dif = abs(srcI - avgI);
            int th = weight*varMat.at<float>(h, w);

            if (dif > th) {
                dstMat.at<uchar>(h, w) = 255;
            }
            else {
                dstMat.at<uchar>(h, w)=0;
            }
        }
    }
    return 0;
}

int calcGaussianBackground(vector<Mat> srcMats, Mat & avgMat, Mat &varMat)
{
    int rows = srcMats[0].rows;
    int cols = srcMats[0].cols;

    for (int h = 0; h < rows; h++)
    {
        for (int w = 0; w < cols; w++)
        {
            int sum=0;
            float var=0;
            for (int i = 0; i < srcMats.size(); i++) {
                sum += srcMats[i].at<uchar>(h, w);
            }
            avgMat.at<uchar>(h, w)=sum / srcMats.size();
            for (int i = 0; i < srcMats.size(); i++) {
                var += pow((srcMats[i].at<uchar>(h, w) - avgMat.at<uchar>(h, w)), 2);
            }
            varMat.at<float>(h, w) = var / srcMats.size();
        }
    }
    return 0;
 }

int main()
{
    VideoCapture capVideo(0);

    vector<Mat> srcMats;
    float wVar = 1;

    int cnt = 0;
    Mat frame;
    Mat avgMat;
    Mat varMat;
    Mat dstMat;

    while (true)
    {
        capVideo >> frame;
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        if (cnt < 200) {
            srcMats.push_back(frame);
            if (cnt == 0) {
                cout << "reading frame " << endl;
            }
        }
        else if (cnt == 200) {
            avgMat.create(frame.size(),CV_8UC1);
            varMat.create(frame.size(),CV_32FC1);
            cout << "calculating background models" << endl;
            calcGaussianBackground(srcMats,avgMat,varMat);
        }
        else {
            dstMat.create(frame.size(), CV_8UC1);
            gaussianThreshold(frame, avgMat, varMat, wVar, dstMat);
            imshow("result",dstMat);
            imshow("frame",frame);
            waitKey(30);
        }
        cnt++;
    }
    return 0;
}
