//
// Created by yh on 18-11-12.
//
#include <iostream>
#include <string>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGBA PointT; //定义空间点的数据类型
typedef pcl::PointCloud<PointT>PointCloud; //点云集

const double camera_factor=1000;
const double camera_cx=325.5;
const double camera_cy=253.5;
const double camera_fx=518.0;
const double camera_fy=519.0;

int main(int argc,char**argv)
{
    cv::Mat rgb,depth;
    rgb=cv::imread("/home/yh/下载/rgb.png");
    depth=cv::imread("/home/yh/下载/depth.png",-1);
    PointCloud ::Ptr cloud(new PointCloud); //智能指针 ：：作用域 new返回的是指针
    //遍历深度图
    for(int m=0;m<depth.rows;m++)
        for (int n = 0; n <depth.cols ; n++)
        {
            ushort d=depth.ptr<ushort>(m)[n];  //ptr是opencv中的函数，返回地m行的头指针，加上n表示指向的数据
                     if(d==0)
                         continue;
                     PointT p;
                     p.z=double(d)/camera_factor;
                     p.y=(m-camera_cy)*p.z/camera_fy;
                     p.x=(n-camera_cx)*p.z/camera_fx;  //m=v?  n=u?? 把深度图的起点看成了(0,0）
                             p.b=rgb.ptr<uchar>(m) [n*3] ;
                             p.g=rgb.ptr<uchar>(m)[n*3+1];
                             p.r=rgb.ptr<uchar>(m)[n*3+2];
                             cloud->points.push_back(p);
        }

        cloud->height=1;
    cloud->width=cloud->points.size();
    cout<<"cloud size="<<cloud->points.size()<<endl;
    cloud->is_dense= false;
    pcl::io::savePCDFile("./pointcloud.pcd",*cloud);
    cloud->points.clear();
    cout<<"point cloud save."<<endl;
    return 0;


}
