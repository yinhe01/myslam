//
// Created by yh on 18-11-21.
//
#include<iostream>
#include <sstream>
#include <fstream>

using namespace std;

#include "slambase.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

//把g2o的定义放在前面
typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;


//给定index读取一帧图像
FRAME readFrame(int index,ParameterReader& pd);
//计算一个运动的大小
double normofTransform(cv::Mat rvec,cv::Mat tvec);
//检测两帧，结果定义
enum CHECK_RESULT {NOT_MATCHED=0,TOO_FAR_AWAY,TOO_CLOSE,KEYFRAME};
//检测关键帧
CHECK_RESULT checkKeyframes(FRAME& f1,FRAME& f2,g2o::SparseOptimizer& opti, bool is_loops= false);
//检测近距离回环
void checkNearbyLoops(vector<FRAME>& frames,FRAME& currFrame,g2o::SparseOptimizer& opti);
//检测随机回环
void checkRandomLoops(vector<FRAME>& frames,FRAME& currFrame,g2o::SparseOptimizer& opti);




int main(int argc,char** argv)
{
    ParameterReader pd;
    int startIndex=atoi(pd.getData("start_index").c_str());
    int endIndex=atoi(pd.getData("end_index").c_str());

    vector<FRAME> keyframes;  //关键帧集合
    //初始化...
    cout<<"Initializing"<<endl;
    int currIndex=startIndex;
    FRAME currFrame=readFrame(currIndex,pd);

    string detector=pd.getData("detector");
    string descriptor=pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera=getDefaultCamera();
    computeKeyPointsAndDesp(currFrame,detector,descriptor);
    PointCloud::Ptr cloud=image2PointCloud(currFrame.rgb,currFrame.depth,camera);

    //g2o初始化
    //初始化求解器
    SlamLinearSolver* linearSolver=new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);
    SlamBlockSolver* blockSolver=new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(blockSolver);

    g2o::SparseOptimizer globalOptimizer;  //最后要使用的优化器
    globalOptimizer.setAlgorithm(solver);

    //禁止输出调试信息
    globalOptimizer.setVerbose(false);



    //添加顶点
    g2o::VertexSE3* v=new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity());
    v->setFixed(true);
    globalOptimizer.addVertex(v);

    keyframes.push_back(currFrame);

    double keyframe_threshold=atof(pd.getData("keyframe_threshold").c_str());
    bool check_loop_closure=pd.getData("check_loop_closure")==string("yes");

    for ( currIndex=startIndex+1; currIndex<endIndex; currIndex++ )
    {
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = readFrame( currIndex,pd ); // 读取currFrame
        computeKeyPointsAndDesp( currFrame, detector, descriptor ); //提取特征
        CHECK_RESULT result = checkKeyframes( keyframes.back(), currFrame, globalOptimizer ); //匹配该帧与keyframes里最后一帧
        switch (result) // 根据匹配结果不同采取不同策略
        {
            case NOT_MATCHED:
                //没匹配上，直接跳过
                cout<<RED"Not enough inliers."<<endl;
                break;
            case TOO_FAR_AWAY:
                // 太近了，也直接跳
                cout<<RED"Too far away, may be an error."<<endl;
                break;
            case TOO_CLOSE:
                // 太远了，可能出错了
                cout<<RESET"Too close, not a keyframe"<<endl;
                break;
            case KEYFRAME:
                cout<<GREEN"This is a new keyframe"<<endl;
                // 不远不近，刚好
                /**
                 * This is important!!
                 * This is important!!
                 * This is important!!
                 * (very important so I've said three times!)
                 */
                // 检测回环
                if (check_loop_closure)
                {
                    checkNearbyLoops( keyframes, currFrame, globalOptimizer );
                    checkRandomLoops( keyframes, currFrame, globalOptimizer );
                }
                keyframes.push_back( currFrame );

                break;
            default:
                break;
        }

    }

    //优化
    cout<<"optimizing pose graph,vertices:"<<globalOptimizer.vertices().size()<<endl;
    globalOptimizer.save("/home/yh/myslam/data/result_before_g2o.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100);
    globalOptimizer.save("/home/yh/myslam/data/result_after_g2o.g2o");
    cout<<"Optimization done."<<endl;

    //拼接点云地图          ?????????????????????????????????????????????????????????????????????????
    cout<<"saving the point cloud..."<<endl;
    PointCloud::Ptr output(new PointCloud()); //全局地图
    PointCloud::Ptr tmp(new PointCloud());
    pcl::VoxelGrid<PointT> voxel;
    pcl::PassThrough<PointT>pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0,4.0);
    double gridsize=atof(pd.getData("voxel_grid").c_str());
    voxel.setLeafSize(gridsize,gridsize,gridsize);
    pcl::visualization::CloudViewer viewer("map");
    for (size_t i=0;i<keyframes.size();i++)
    {
        g2o::VertexSE3* vertex= dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID)); //这句没看懂
        Eigen::Isometry3d pose=vertex->estimate();
        PointCloud::Ptr newCloud=image2PointCloud(keyframes[i].rgb,keyframes[i].depth,camera);
        //点云滤波
        voxel.setInputCloud(newCloud);
        voxel.filter(*tmp);
        pass.setInputCloud(tmp);
        pass.filter(*newCloud);
        pcl::transformPointCloud(*newCloud,*tmp,pose.matrix());
        *output+=*tmp;
        newCloud->clear();
    }

    voxel.setInputCloud(output);
    voxel.filter(*tmp);
    pcl::io::savePCDFile("/home/yh/myslam/data/result.pcd",*tmp);
    cout<<"Final map is saved."<<endl;
    globalOptimizer.clear();
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
    f.frameID=index;
    return f;      //都是骚操作
}

double normofTransform(cv::Mat rvec,cv::Mat tvec)
{
    return fabs(min(cv::norm(rvec),2*M_PI-cv::norm(rvec)))+fabs(cv::norm(tvec)); //就是个公式
}

CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    static double max_norm = atof( pd.getData("max_norm").c_str() );
    static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    static double max_norm_lp = atof( pd.getData("max_norm_lp").c_str() );
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    // 比较f1 和 f2
    RESULT_OF_PNP result = estimateMotion( f1, f2, camera );
    if ( result.inliers < min_inliers ) //inliers不够，放弃该帧
        return NOT_MATCHED;
    // 计算运动范围是否太大
    double norm = normofTransform(result.rvec, result.tvec);
    if ( is_loops == false )
    {
        if ( norm >= max_norm )
            return TOO_FAR_AWAY;   // too far away, may be error
    }
    else
    {
        if ( norm >= max_norm_lp)
            return TOO_FAR_AWAY;
    }

    if ( norm <= keyframe_threshold )
        return TOO_CLOSE;   // too adjacent frame
    // 向g2o中增加这个顶点与上一帧联系的边
    // 顶点部分
    // 顶点只需设定id即可
    if (is_loops == false)
    {
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId( f2.frameID );
        v->setEstimate( Eigen::Isometry3d::Identity() );
        opti.addVertex(v);
    }
    // 边部分
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    // 连接此边的两个顶点id
    edge->setVertex( 0, opti.vertex(f1.frameID ));
    edge->setVertex( 1, opti.vertex(f2.frameID ));
    edge->setRobustKernel( new g2o::RobustKernelHuber() );
    // 信息矩阵
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;
    // 也可以将角度设大一些，表示对角度的估计更加准确
    edge->setInformation( information );
    // 边的估计即是pnp求解之结果
    Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
    // edge->setMeasurement( T );
    edge->setMeasurement( T.inverse() );
    // 将此边加入图中
    opti.addEdge(edge);
    return KEYFRAME;
}


void checkNearbyLoops(vector<FRAME>& frames,FRAME& currFrame,g2o::SparseOptimizer& opti)
{
    static ParameterReader pd;
    static int nearby_loops=atoi(pd.getData("nearby_loops").c_str());
    if (frames.size()<nearby_loops)
        //没有足够的关键帧，检测所有
    {
        for(size_t i=0;i<frames.size();i++)
        {
            checkKeyframes(frames[i],currFrame,opti, true);    //只添加边就行了，因为他已经是个关键帧了，顶点已经存在
        }
    }
    else
    {
        for (size_t i=frames.size()-nearby_loops;i<frames.size();i++)
        {
            checkKeyframes(frames[i],currFrame,opti, true);       //不检查距离，只检查内点数够不够，够的话，添加一条边进来
        }
    }

}


void checkRandomLoops(vector<FRAME>& frames,FRAME& currFrame,g2o::SparseOptimizer& opti) {
    static ParameterReader pd;
    static int random_loops = atoi(pd.getData("random_loops").c_str());
    srand((unsigned int) time(NULL));  //随机抽取一些帧  初始化随机数发生器
    if (frames.size() <= random_loops) {
        for (size_t i = 0; i < frames.size(); i++) {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    } else {
        for (size_t i = 0; i < random_loops; i++) {
            int index = rand() % frames.size();  //产生一个0-frames.size()的随机数
            checkKeyframes(frames[index], currFrame, opti, true);
        }
    }
}