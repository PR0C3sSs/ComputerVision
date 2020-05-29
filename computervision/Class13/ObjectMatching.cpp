#include <stdio.h>
#include <opencv2/opencv.hpp>
#define threshold 10000
using namespace std;
using namespace cv;

int myHOG(Mat src, float * hist,int nX, int nY ,int dimension, int blocksize);
float dist(float * Hist1, float * Hist2, int bins);

int main(){
    Mat temp = imread("/Users/joey/Desktop/template.png",0);
    Mat src = imread("/Users/joey/Desktop/img.png",0);
    
    Mat gx, gy;
    Mat mag, angle;
    //计算模版的梯度和角度
    Sobel(temp, gx, CV_32F, 1, 0, 1);
    Sobel(temp, gy, CV_32F, 0, 1, 1);
    cartToPolar(gx, gy, mag,angle ,true);
    
    int dimension = 8;
    int blocksize = 16;
    int nX = temp.cols / blocksize;
    int nY = temp.rows / blocksize;

    int bins = nX * nY * dimension;
    
    int minX = 0, minY = 0, minDis = 30000;
    
    for (int x = 0; x < (src.rows - temp.rows); x++) {
        for (int y = 0; y < (src.cols - temp.cols); y++) {
            Mat roi(src, Rect(y, x, temp.cols, temp.rows));
            
            float * temp_hist = new float[bins];
            memset(temp_hist, 0, sizeof(float)*bins);
            float * roi_hist = new float[bins];
            memset(roi_hist, 0, sizeof(float)*bins);
            
            int reCode = 0;
            
            reCode = myHOG(temp, temp_hist, nX, nY,dimension, blocksize);
            reCode = myHOG(roi, roi_hist, nX, nY,dimension, blocksize);

            if (reCode != 0) {
                return -1;
            }
            
            float dis = dist(temp_hist, roi_hist, bins);
            if (dis < minDis) {
                minDis = dis;
                minX = x;
                minY = y;
            }
        }
    }
    
    cout << "minDis = "<< minDis << endl;
    cout << "minX = "<< minX << endl;
    cout << "minY = "<< minY << endl;
    
    rectangle(src,Rect(minX,minY,temp.cols,temp.rows),Scalar(255,0,0),1,1,0);
    
    imshow("template", temp);
    imshow("src", src);
    
    waitKey(0);
    return 0;
}

int myHOG(Mat src, float * hist,int nX, int nY ,int dimension, int blocksize){

    Mat gx, gy;
    Mat mag, angle;
    Sobel(src, gx, CV_32F, 1, 0, 1);
    Sobel(src, gy, CV_32F, 0, 1, 1);
    cartToPolar(gx, gy, mag, angle, true);
    
    int level = 360 / dimension;
    for (int i = 0; i < nY; i++) {
        for (int j = 0; j < nX; j++) {
            Rect block(j*blocksize,i*blocksize,blocksize,blocksize);
            Mat Current = src(block);
            Mat Gredient = mag(block);
            Mat Angle = angle(block);
            
            for (int n = 0; n < blocksize; n++) {
                for (int m = 0; m < blocksize; m++) {
                    //计算梯度
                    int position = Angle.at<float>(n, m) / level;
                    hist[((i*nX + j) * dimension)+position] += Gredient.at<float>(n, m);
                }
            }
        
        }
    }
    return 0;
}

float dist(float * Hist1, float * Hist2, int bins)
{
    float sum = 0;
    for (int i = 0; i < bins; i++) {
        sum += pow((Hist1[i] - Hist2[i]),2);
    }
    return sqrt(sum);
}
