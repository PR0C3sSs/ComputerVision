#include <stdio.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

//参数设置
#define YOLOV3_VIDEO        "/Users/joey/Downloads/本来是一场很严肃的车祸，看完监控，你们忍住“别笑”.166491947.flv"
#define OPENPOSE_VIDEO        "/Users/joey/Downloads/蔡徐坤打篮球原视频.106875795.flv"

using namespace cv;
using namespace std;
using namespace dnn;

//通过非极大值抑制去掉置信度较低的bouding box
void postprocess(Mat& frame, vector<Mat>& outs);

// 获得输出名字
vector<String> getOutputsNames(const dnn::Net& net);

//绘制检测结果
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

int yoloV3();

int openpose();

vector<string> classes;

float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image


// key point 连接表, [model_id][pair_id][from/to]
// 详细解释见
// https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md

int main(){
    //开始计时
    double start = static_cast<double>(getTickCount());
    
    yoloV3();
    
    openpose();

    //结束计时
    double time = ((double)getTickCount() - start) / getTickFrequency();
    //显示时间
    cout << "processing time:" << time / 1000 << "ms" << endl;

    //等待键盘响应，按任意键结束程序
    system("pause");
    return 0;
}

int POSE_PAIRS[3][20][2] = {
    {   // COCO body
        { 1,2 },{ 1,5 },{ 2,3 },
        { 3,4 },{ 5,6 },{ 6,7 },
        { 1,8 },{ 8,9 },{ 9,10 },
        { 1,11 },{ 11,12 },{ 12,13 },
        { 1,0 },{ 0,14 },
        { 14,16 },{ 0,15 },{ 15,17 }
    },
    {   // MPI body
        { 0,1 },{ 1,2 },{ 2,3 },
        { 3,4 },{ 1,5 },{ 5,6 },
        { 6,7 },{ 1,14 },{ 14,8 },{ 8,9 },
        { 9,10 },{ 14,11 },{ 11,12 },{ 12,13 }
    },
    {   // hand
        { 0,1 },{ 1,2 },{ 2,3 },{ 3,4 },         // thumb
        { 0,5 },{ 5,6 },{ 6,7 },{ 7,8 },         // pinkie
        { 0,9 },{ 9,10 },{ 10,11 },{ 11,12 },    // middle
        { 0,13 },{ 13,14 },{ 14,15 },{ 15,16 },  // ring
        { 0,17 },{ 17,18 },{ 18,19 },{ 19,20 }   // small
    } };

vector<String> getOutputsNames(const dnn::Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

//非极大值抑制，去除置信度较小的检测结果
void postprocess(Mat& frame, vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        //
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;

            //获得得分最高的结果的分值和位置
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    //非极大值抑制，去除置信度较小的检测结果
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}

//检测结果绘制
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //绘制检测框
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

    //获得识别结果的类名称，以及置信度
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    else
    {
        cout << "classes is empty..." << endl;
    }

    //绘制标签
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseLine);
    top = max(top, labelSize.height);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255),1);
}

//opencv 调用 yolov3 demo
int yoloV3()
{

    VideoCapture cap(YOLOV3_VIDEO);

    if (!cap.isOpened())return -1;


    //coco数据集的名称文件，80类
    string classesFile= "/Users/joey/Documents/Code/XCode/computervision/computervision/Class15/coco.names";
    //yolov3网络模型文件
    String yolov3_model = "/Users/joey/Documents/Code/XCode/computervision/computervision/Class15/yolov3.cfg";
    //权重
    String weights = "/Users/joey/Documents/Code/XCode/computervision/computervision/Class15/yolov3.weights";

    //将coco.names中的80类名称转换为vector形式
    ifstream classNamesFile(classesFile.c_str());
    if (classNamesFile.is_open())
    {
        string className = "";
        // getline (istream&  is, string& str)
        //is为输入，从is中读取读取的字符串保存在string类型的str中，如果没有读入字符返回false，循环结束
        while (getline(classNamesFile, className)) {
            classes.push_back(className);
        }
    }
    else {
        cout << "can not open classNamesFile" << endl;
    }

    dnn::Net net= dnn::readNetFromDarknet(yolov3_model, weights);

    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat frame1;

    while (1)
    {
        cap >> frame1;

        if (frame1.empty()) {
            cout << "frame is empty!!!" << endl;
            return -1;
        }

        //创建yolo输入数据
        Mat blob;
        dnn::blobFromImage(frame1, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

        //输入网络
        net.setInput(blob);

        //定义输出结果保存容器
        vector<Mat> outs;
        //前向传输获得结果
        net.forward(outs, getOutputsNames(net));

        //后处理，非极大值抑制，绘制检测框
        postprocess(frame1, outs);

        imshow("yolo", frame1);

        if (waitKey(10) == 27)
        {
            break;
        }
    }

    return 0;

}

int openpose()
{

    //读入网络模型和权重文件
    String modelTxt = "/Users/joey/Documents/Code/XCode/computervision/computervision/Class15/openpose_pose_coco.prototxt";
    String modelBin = "/Users/joey/Documents/Code/XCode/computervision/computervision/Class15/caffe_models/pose/coco/pose_iter_440000.caffemodel";

    dnn::Net net = dnn::readNetFromCaffe(modelTxt, modelBin);

    int W_in = 368;
    int H_in = 368;
    float thresh = 0.1;

    VideoCapture cap;
    cap.open(OPENPOSE_VIDEO);

    if (!cap.isOpened())return -1;

    while (1) {

        Mat frame2;

        cap >> frame2;

        if (frame2.empty()) {
            cout << "frame is empty!!!" << endl;
            return -1;
        }

        //创建输入
        Mat inputBlob = blobFromImage(frame2, 1.0 / 255, Size(W_in, H_in), Scalar(0, 0, 0), false, false);

        //输入
        net.setInput(inputBlob);

        //得到网络输出结果，结果为热力图
        Mat result = net.forward();

        int midx, npairs;
        int H = result.size[2];
        int W = result.size[3];

        //得到检测结果的关键点点数
        int nparts = result.size[1];
        

        // find out, which model we have
        //判断输出的模型类别
        if (nparts == 19)
        {   // COCO body
            midx = 0;
            npairs = 17;
            nparts = 18; // skip background
        }
        else if (nparts == 16)
        {   // MPI body
            midx = 1;
            npairs = 14;
        }
        else if (nparts == 22)
        {   // hand
            midx = 2;
            npairs = 20;
        }
        else
        {
            cerr << "there should be 19 parts for the COCO model, 16 for MPI, or 22 for the hand one, but this model has " << nparts << " parts." << endl;
            return (0);
        }

        // 获得身体各部分坐标
        vector<Point> points(22);
        for (int n = 0; n < nparts; n++)
        {
            // Slice heatmap of corresponding body's part.
            Mat heatMap(H, W, CV_32F, result.ptr(0, n));
            // 找到最大值的点
            Point p(-1, -1), pm;
            double conf;
            minMaxLoc(heatMap, 0, &conf, 0, &pm);
            //判断置信度
            if (conf > thresh) {
                p = pm;
            }
            points[n] = p;
        }

        //连接身体各个部分，并且绘制
        float SX = float(frame2.cols) / W;
        float SY = float(frame2.rows) / H;
        for (int n = 0; n < npairs; n++)
        {
            Point2f a = points[POSE_PAIRS[midx][n][0]];
            Point2f b = points[POSE_PAIRS[midx][n][1]];

            //如果前一个步骤没有找到相应的点，则跳过
            if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
                continue;

            // 缩放至图像的尺寸
            a.x *= SX; a.y *= SY;
            b.x *= SX; b.y *= SY;

            //绘制
            line(frame2, a, b, Scalar(0, 200, 0), 2);
            circle(frame2, a, 3, Scalar(0, 0, 200), -1);
            circle(frame2, b, 3, Scalar(0, 0, 200), -1);
        }

        imshow("openpose",frame2);

        waitKey(30);

    }



    return 0;
}
