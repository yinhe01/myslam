//
// Created by yh on 18-11-13.
//
#include <opencv2/features2d/features2d.hpp>
#include "slambase.h"
PointCloud::Ptr image2PointCloud(cv::Mat& rgb,cv::Mat& depth,CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr cloud(new PointCloud);

    for (int m=0;m<depth.rows;m++)
        for(int n=0;n<depth.cols;n++)
        {
            ushort d=depth.ptr<ushort>(m)[n];
            if (d==0)
                continue;
            PointT p;
            p.z=double(d)/camera.scale;
            p.x=(n-camera.cx)*p.z/camera.fx;
            p.y=(m-camera.cy)*p.z/camera.fy;
            p.b=rgb.ptr<uchar>(m)[n*3];
            p.g=rgb.ptr<uchar>(m)[n*3+1];
            p.r=rgb.ptr<uchar>(m)[n*3+2];
            cloud->points.push_back(p);
        }
        cloud->height=1;
    cloud->width=cloud->points.size();
    cloud->is_dense=false;
    return cloud;
}

cv::Point3f point2dTo3d(cv::Point3f& point,CAMERA_INTRINSIC_PARAMETERS& camera)
{
    cv::Point3f p;
    p.z=double(point.z)/camera.scale;
    p.x=(point.x-camera.cx)*p.z/camera.fx;
    p.y=(point.y-camera.cy)*p.z/camera.fy;
    return p;
}


void computeKeyPointsAndDesp(Frame& frame,string detector,string descriptor)
{
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;
    _detector=cv::FeatureDetector::create( detector.c_str());
    _descriptor=cv::DescriptorExtractor::create(descriptor.c_str());
    if (!_detector || !_descriptor)
    {
                cerr<<"Unknown detector or discriptor type !"<<detector<<","<<descriptor<<endl;
               return;
    }

    _detector->detect(frame.rgb,frame.kp);
    _descriptor->compute(frame.rgb,frame.kp,frame.desp);
    return;
}


Result_of_PnP estimateMotion(Frame& frame1,Frame& frame2,CAMERA_INTRINSIC_PARAMETERS& camera)
{
    static ParameterReader pd;
    vector<cv::DMatch>matches;
    cv::FlannBasedMatcher matcher;
    matcher.match(frame1.desp,frame2.desp,matches);
    cout<<"find total"<<matches.size()<<"matches."<<endl;
    vector<cv::DMatch> goodMatches;
    double minDis=9999.0;
    for(size_t i=0;i<matches.size();i++)
    {
        if(matches[i].distance<minDis)
            minDis=matches[i].distance;
    }

    for (size_t i=0;i<matches.size();i++)
    {
        if(matches[i].distance<10*minDis)
        {
            goodMatches.push_back(matches[i]);
        }
    }

    cout<<"good matches:"<<goodMatches.size()<<endl;

    //第一帧的三维点
    vector<cv::Point3f> pts_obj;
    //第二帧的图像点
    vector<cv::Point2d> pts_img;

    for(size_t i=0;i<goodMatches.size();i++)
    {
        cv::Point2f p=frame1.kp[goodMatches[i].queryIdx].pt;
        cv::Point2f q=frame2.kp[goodMatches[i].trainIdx].pt;
        ushort d=frame1.depth.ptr<ushort >(int(p.y))[int(p.x)];
        if(d==0)
        {
            continue;
        }
        pts_img.push_back(q);

        //2D TO 3D
        cv::Point3f pt(p.x,p.y,d);
        cv::Point3f pd=point2dTo3d(pt,camera);
        pts_obj.push_back(pd);
    }

    double camera_matrix_data[3][3] = {
            {camera.fx, 0, camera.cx},
            {0, camera.fy, camera.cy},
            {0, 0, 1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;

    cv::solvePnPRansac(pts_obj,pts_img,cameraMatrix,cv::Mat(),rvec,tvec, false,100,1.0,100,inliers);
    Result_of_PnP result;
    result.inliers=inliers.rows;
    result.rvec=rvec;
    result.tvec=tvec;
    return result;

}