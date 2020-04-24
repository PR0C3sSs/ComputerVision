#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <math.h>

using namespace cv;
using namespace std;

int myHOG(Mat src, float * hist,int nX, int nY ,int dimension, int blocksize);
float dist(float * Hist1, float * Hist2, int bins);

int main(){
    Mat srcMat = imread("/Users/joey/Documents/Code/XCode/computervision/computervision/Class09/hogTemplate.jpg",0);
    Mat img1Mat = imread("/Users/joey/Documents/Code/XCode/computervision/computervision/Class09/img1.jpg",0);
    Mat img2Mat = imread("/Users/joey/Documents/Code/XCode/computervision/computervision/Class09/img2.jpg",0);

    Mat gx, gy;
    Mat mag, angle;
    //计算梯度和角度
    Sobel(srcMat, gx, CV_32F, 1, 0, 1);
    Sobel(srcMat, gy, CV_32F, 0, 1, 1);
    //x方向梯度，y方向梯度，梯度，角度，决定输出弧度or角度
    cartToPolar(gx, gy, mag,angle ,true);

    int dimension = 8;
    int blocksize = 16;
    int nX = srcMat.cols / blocksize;
    int nY = srcMat.rows / blocksize;

    int bins = nX * nY * dimension;
    
    float * src_hist = new float[bins];
    memset(src_hist, 0, sizeof(float)*bins);
    float * img1_hist = new float[bins];
    memset(img1_hist, 0, sizeof(float)*bins);
    float * img2_hist = new float[bins];
    memset(img2_hist, 0, sizeof(float)*bins);
    
    int reCode = 0;
    //计算三张输入图片的HOG
    reCode = myHOG(srcMat, src_hist, nX, nY,dimension, blocksize);
    reCode = myHOG(img1Mat, img1_hist, nX, nY,dimension, blocksize);
    reCode = myHOG(img2Mat, img2_hist, nX, nY,dimension, blocksize);

    if (reCode != 0) {
        return -1;
    }
    
    float dis1 = dist(src_hist, img1_hist, bins);
    float dis2 = dist(src_hist, img2_hist, bins);
    
    cout<<"dis1 = "<<dis1<<endl<<"dis2 = "<< dis2<<endl;
    
    cout << "the more similar one is : ";
    if(dis1<=dis2){
        cout << "img1" << endl;
    }else{
        cout << "img2" <<endl;
        
    }

    delete[] src_hist;
    delete[] img1_hist;
    delete[] img2_hist;

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
