//
// Created by yh on 18-11-21.
//

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

for (currIndex=startIndex+1;currIndex<endIndex;currIndex++)
{
cout<<"reading flie"<<currIndex<<endl;
FRAME currFrame=readFrame(currIndex,pd);
computeKeyPointsAndDesp(currFrame,detector,descriptor);

CHECK_RESULT result=checkKeyframes(keyframes.back(),currFrame,globalOptimizer);//使用keyframes.back()代替了lastframe
//其中计算了PnP
switch(result)
{
case NOT_MATCHED:
cout << "Not enough inliers" << endl;
break;

case TOO_FAR_AWAY:
cout << "Too far away.maybe an error" << endl;
break;

case TOO_CLOSE:
cout << "Too close,not a keyframe" << endl;
break;

case KEYFRAME:
cout <<"This is a new keyframe" << endl;

//检测回环
if (check_loop_closure) {
checkNearbyLoops(keyframes, currFrame, globalOptimizer);
checkRandomLoops(keyframes, currFrame, globalOptimizer);
}
keyframes.push_back(currFrame);
break;
default:
break;
}
}


CHECK_RESULT checkKeyframes(FRAME& f1,FRAME& f2,g2o::SparseOptimizer& opti, bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers=atoi(pd.getData("min_inliers").c_str());
    static double max_norm=atof(pd.getData("max_norm").c_str());
    static double keyframe_threshold=atof(pd.getData("keyframe_threshold").c_str());
    static double max_norm_lp=atof(pd.getData("max_norm_lp").c_str());
    static CAMERA_INTRINSIC_PARAMETERS camera=getDefaultCamera();
    //static g2o::RobustKernel* robustKernel=g2o::RobustKernelFactory::instance()->construct("Cauchy");
    // 比较f1和f2
    RESULT_OF_PNP result=estimateMotion(f1,f2,camera);
    if(result.inliers<min_inliers)
        return NOT_MATCHED;
    double norm=normofTransform(result.rvec,result.tvec);
    if(is_loops== false)
    {
        if (norm>=max_norm)
            return TOO_FAR_AWAY;
    }
    else
    {
        if (norm>=max_norm_lp)
            return TOO_FAR_AWAY;
    }

    if (norm<=keyframe_threshold)
        return TOO_CLOSE;


    //g2o部分
    if (is_loops== false)
    {
        g2o::VertexSE3 *v=new g2o::VertexSE3();
        v->setId(f2.frameID);
        v->setEstimate(Eigen::Isometry3d::Identity());
        opti.addVertex(v);
    }

    g2o::EdgeSE3* edge=new g2o::EdgeSE3();
    edge->vertices() [0]=opti.vertex(f1.frameID);
    edge->vertices() [1]=opti.vertex(f2.frameID);
    edge->setRobustKernel(new g2o::RobustKernelHuber());
    //信息矩阵
    Eigen::Matrix<double,6,6> information=Eigen::Matrix<double,6,6>::Identity();
    //信息矩阵表示我们对边的精度的预先估计
    information(0,0)=information(1,1)=information(2,2)=information(3,3)=information(4,4)=information(5,5)=100;
    edge->setInformation(information);
    Eigen::Isometry3d T=cvMat2Eigen(result.rvec,result.tvec);
    edge->setMeasurement(T.inverse());
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
            checkKeyframes(frames[i],currFrame,opti, true);
        }
    }

}


void checkRandomLoops(vector<FRAME>& frames,FRAME& currFrame,g2o::SparseOptimizer& opti)
{
    static ParameterReader pd;
    static int random_loops=atoi(pd.getData("random_loops").c_str());
    srand((unsigned int) time(NULL));  //随机抽取一些帧  初始化随机数发生器
    if(frames.size()<=random_loops)
    {
        for (size_t i=0;i<frames.size();i++)
        {
            checkKeyframes(frames[i],currFrame,opti, true);
        }
    }
    else
    {
        for (size_t i=0;i<random_loops;i++)
        {
            int index=rand()%frames.size();  //产生一个0-frames.size()的随机数
            checkKeyframes(frames[index],currFrame,opti, true);
        }
    }