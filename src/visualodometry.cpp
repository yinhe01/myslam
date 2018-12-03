//
// Created by yh on 18-11-19.
//

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slambase.h"

//读取一帧数据
FRAME readFrame(int index,ParameterReader& pd);  //函数声明

//度量运动的大小
double normofTransform(cv::Mat rvec,cv::Mat tvec);


int main(int argv,char** )
{
    ParameterReader pd;
    int startIndex=atoi(pd.getData("start_index").c_str());
    int endIndex=atoi(pd.getData("end_index").c_str());

    //初始化
    cout<<"initializing..."<<endl;
    int currIndex=startIndex;
    FRAME lastFrame=readFrame(currIndex,pd); //上一阵数据

            //在 lastFrame和currFrame之间进行比较
    string detector=pd.getData("detector");
    string descriptor=pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera=getDefaultCamera();
    computeKeyPointsAndDesp(lastFrame,detector,descriptor);
    PointCloud::Ptr cloud=image2PointCloud(lastFrame.rgb,lastFrame.depth,camera);
    pcl::visualization::CloudViewer viewer("viewer");

    //是否显示点云
    bool visualize=pd.getData("visualize_pointcloud")==string("yes");
    int min_inliers=atoi(pd.getData("min_inliers").c_str()); //最小内点
    double max_norm=atof(pd.getData("max_norm").c_str());   //最大变换距离
    for(currIndex=startIndex+1;currIndex<endIndex;currIndex++)
    {
        cout<<"reading files"<<currIndex<<endl;
        FRAME currFrame=readFrame(currIndex,pd);
        computeKeyPointsAndDesp(currFrame,detector,descriptor);
        RESULT_OF_PNP result=estimateMotion(lastFrame,currFrame,camera);
        if (result.inliers<min_inliers)
            continue;   //放弃该帧
        double norm=normofTransform(result.rvec,result.tvec);
        if (norm>=max_norm)
            continue;   //放弃该帧

        Eigen::Isometry3d T=cvMat2Eigen(result.rvec,result.tvec);
        cout<<"T="<<T.matrix()<<endl;

        //合并点云
        cloud=joinPointCloud(cloud,currFrame,T,camera);

        if (visualize== true)
            viewer.showCloud(cloud);

        lastFrame=currFrame; //实时更新上一帧
    }
    pcl::io::loadPCDFile("/home/yinhe/myslam/data/result.pcd",*cloud);
    return 0;

}

FRAME readFrame(int index,ParameterReader& pd)
{
    FRAME f;
    string rgbDir = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");
    string depthExt = pd.getData("depth_extension");
    string rgbExt = pd.getData("rgb_extension");

    stringstream ss;   //一个用于c++风格的字符串输入输出类
    ss << rgbDir << index << rgbExt;   //输入一些东西
    string filename;
    ss>>filename;   //  把这些东西转化成文件名
    f.rgb=cv::imread(filename);

    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;
    f.depth=cv::imread(filename,-1);
    return f;      //都是骚操作
}

double normofTransform(cv::Mat rvec,cv::Mat tvec)
{
    return fabs(min(cv::norm(rvec),2*M_PI-cv::norm(rvec)))+fabs(cv::norm(tvec)); //就是个公式
}

