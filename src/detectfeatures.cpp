//
// Created by yh on 18-11-13.
//
#include <iostream>
#include "slambase.h"
using namespace std;

//opencv特征检测模块
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

int main(int argc,char** argv)
{
    cv::Mat rgb1=cv::imread("/home/yh/下载/1.png");
    cv::Mat rgb2=cv::imread("/home/yh/下载/2.png");
    cv::Mat depth1=cv::imread("/home/yh/下载/depth1.png",-1);
    cv::Mat depth2=cv::imread("/home/yh/下载/depth2.png",-1);

    //声明特征提取器和描述子提取器
            cv::Ptr<cv::FeatureDetector>_detector;
            cv::Ptr<cv::DescriptorExtractor>_descriptor;

            //构建提取器
    //构建sift
    _detector=cv::FeatureDetector::create("GridSIFT");
    _descriptor=cv::DescriptorExtractor::create("SIFT");
    //关键点
            vector<cv::KeyPoint>kp1,kp2;
            //提取关键点
    _detector->detect(rgb1,kp1);
    _detector->detect(rgb2,kp2);
    cout<<"Ket points size of two images"<<kp1.size()<<","<<kp2.size()<<endl;
//可视化关键点
    cv::Mat imgShow;
    cv::drawKeypoints(rgb1,kp1,imgShow,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("keypoints",imgShow);
    cv::imwrite("/home/yh/下载/keypoint.png",imgShow);
    cv::waitKey(0);

    //计算描述子
    cv::Mat desp1,desp2;
    _descriptor->compute(rgb1,kp1,desp1);
    _descriptor->compute(rgb2,kp2,desp2);

    //描述子匹配；
    vector<cv::DMatch>matches;
    cv::FlannBasedMatcher matcher;
    matcher.match(desp1,desp2,matches);
    cout<<"FInd total"<<matches.size()<<"matches."<<endl;

    //可视化；显示匹配的特征
    cv::Mat imgMatches;
    cv::drawMatches(rgb1,kp1,rgb2,kp2,matches,imgMatches);
    cv::imshow("matches",imgMatches);
    cv::waitKey(0);

    //筛选匹配点，把距离太大的去掉
    vector<cv::DMatch>goodMatches;
    double minDis=9999;
    for (size_t i=0;i<matches.size();i++)
    {
        if(matches[i].distance<minDis)
            minDis=matches[i].distance;
    }

    for(size_t i=0;i<matches.size();i++)
    {
        if(matches[i].distance<minDis*4)
            goodMatches.push_back(matches[i]);
    }

    //显示good matches
    cout<<"good matches="<<goodMatches.size()<<endl;
    cv::drawMatches(rgb1,kp1,rgb2,kp2,goodMatches,imgMatches);
    cv::imshow("good matches",imgMatches);
    cv::imwrite("/home/yh/下载/good_matches.png",imgMatches);
    cv::waitKey(0);


    //计算图像之间的运动关系`
    //关键函数：cv::solvePnPRansac()

    //准备参数
    CMAERA_INTRINSIC_PARAMETERS C;
    C.cx=325.5;
    C.cy=253.5;
    C.fx=518.0;
    C.fy=519.0;
    C.scale=1000;

    //第一帧的三维点
            vector<cv::Point3f>pts_obj;
            //第二针的图像点
    vector<cv::Point2f>pts_img;

    for (size_t i=0;i<goodMatches.size();i++)
    {
        cv::Point2f p=kp1[goodMatches[i].queryIdx].pt; //没理解啥意思？？？
                ushort d=depth1.ptr<ushort>(int(p.y))[int(p.x)];
                if(d==0)
                {
                    continue;
                }
                pts_img.push_back(cv::Point2f((kp2)[goodMatches[i].trainIdx].pt));

                //将（u,v,d)转化成（x,y,z)
        cv::Point3f pt (p.x,p.y,d);//这是2d特征点
        cv::Point3f pd=point2dTo3d(pt,C);
        pts_obj.push_back(pd);
    }

    double camera_matrix_data[3][3]={{C.fx,0,C.cx},{0,C.fy,C.cy},{0,0,1}};
    cv::Mat cameraMatrix(3,3,CV_64F,camera_matrix_data);
    cv::Mat rvec,tvec,inliers;

    //求解pnp
    cv::solvePnPRansac(pts_obj,pts_img,cameraMatrix,cv::Mat(),rvec,tvec, false,100,1.0,100,inliers);
    cout<<"inlines"<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"T="<<tvec<<endl;

    //画出内点匹配
    vector<cv::DMatch>matchesShow;
    for(size_t i=0;i<inliers.rows;i++)
    {
        matchesShow.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
    }
    cv::drawMatches(rgb1,kp1,rgb2,kp2,matchesShow,imgMatches);
    cv::imshow("inlier matches",imgMatches);
    cv::imwrite("/home/yh/下载/inliners.png",imgMatches);

    cv::waitKey(0);

    return 0;

}
