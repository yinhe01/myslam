//
// Created by yh on 18-11-20.
//

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

#include "slambase.h"

//g2o头文件
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

FRAME readFrame(int index,ParameterReader& pd);
double normofTransform(cv::Mat rvec,cv::Mat tvec);

int main(int argc,char** argv)
{
   ParameterReader pd;
   int startIndex=atoi(pd.getData("start_index").c_str());
   int endIndex=atoi(pd.getData("end_index").c_str());

   //initialize
   cout<<"initializing..."<<endl;
   int currIndex=startIndex;
   FRAME lastFrame=readFrame(currIndex,pd);

   string detector=pd.getData("detector");
   string descriptor=pd.getData("descriptor");
   CAMERA_INTRINSIC_PARAMETERS camera=getDefaultCamera();
   computeKeyPointsAndDesp(lastFrame,detector,descriptor);
   PointCloud::Ptr cloud=image2PointCloud(lastFrame.rgb,lastFrame.depth,camera);

   pcl::visualization::CloudViewer viewer("map");

   //是否显示点云
    bool visualize=pd.getData("visualize_pointcloud")==string("yes");

    int min_inliers=atoi(pd.getData("min_inliers").c_str());
    double max_norm=atof(pd.getData("max_norm").c_str());


    //g2o初始化
    //选择优化方式
    typedef g2o::BlockSolver_6_3 SlamBlockSolver;
    typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

    //初始化求解器
    SlamLinearSolver* linearSolver=new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);
    SlamBlockSolver* blockSolver=new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(blockSolver);

    g2o::SparseOptimizer globalOptimizer;  //最后要使用的优化器
    globalOptimizer.setAlgorithm(solver);

    //禁止输出调试信息
    globalOptimizer.setVerbose(false);

    //向优化器增加第一个顶点
    g2o::VertexSE3* v=new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity());  //设置初始估计
    v->setFixed(true); //第一个顶点固定不用优化
    globalOptimizer.addVertex(v);

    int lastIndex=currIndex;

    for (currIndex=startIndex+1;currIndex<endIndex;currIndex++)
    {
        cout<<"Reading Files"<<currIndex<<endl;
        FRAME currFrame=readFrame(currIndex,pd);
        computeKeyPointsAndDesp(currFrame,detector,descriptor);
        RESULT_OF_PNP result=estimateMotion(lastFrame,currFrame,camera);
        if (result.inliers<min_inliers)
            continue;
        double norm=normofTransform(result.rvec,result.tvec);
        cout<<"norm="<<norm<<endl;
        if (norm>=max_norm)
            continue;
        Eigen::Isometry3d T=cvMat2Eigen(result.rvec,result.tvec);
        cout<<"T="<<T.matrix()<<endl;

        if ( visualize == true )
        {
            cloud = joinPointCloud( cloud, currFrame, T, camera );
            viewer.showCloud( cloud );
        }

        /*
         * 向g2o中增加这个顶点与上一帧联系的变
         */

        //顶点部分  添加顶点
        g2o::VertexSE3* v=new g2o::VertexSE3();
        v->setId(currIndex);
        v->setEstimate(Eigen::Isometry3d::Identity());
        globalOptimizer.addVertex(v);

        //边部分
        g2o::EdgeSE3* edge=new g2o::EdgeSE3();
        //连接此边的两个顶点ID
        edge->vertices() [0]=globalOptimizer.vertex(lastIndex);
        edge->vertices() [1]=globalOptimizer.vertex(currIndex);

        //信息矩阵
        Eigen::Matrix<double,6,6>information=Eigen::Matrix<double,6,6>::Identity();

        information(0,0)=information(1,1)=information(2,2)=information(3,3)=information(4,4)=information(5,5)=100;
        edge->setInformation(information);

        //边的初始估计是T
        edge->setMeasurement(T);

        //将边加入图中
        globalOptimizer.addEdge(edge);

        lastFrame=currFrame;
        lastIndex=currIndex;
    }

    pcl::io::loadPCDFile("/home/yh/myslam/data/result.pcd",*cloud);
    //优化所有的边
    cout<<"optimizing pose graph,vertices:"<<globalOptimizer.vertices().size()<<endl;
    globalOptimizer.save("/home/yh/myslam/data/result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100);  //指定优化步数
    globalOptimizer.save("/home/yh/myslam/data/result_after.g2o");
    cout<<"Optimization Done."<<endl;
    globalOptimizer.clear();
    return  0;
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