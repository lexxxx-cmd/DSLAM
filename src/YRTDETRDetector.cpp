#include "YRTDETRDetector.h"

YDETRDetector::YDETRDetector(const std::string& model_path, const std::string& class_path){
    //好像是禁用优化啥的
    torch::jit::setTensorExprFuserEnabled(false);
    //加载模型
    try{
        mModule = torch::jit::load(model_path);
        mModule.eval();
        // 打印一下参数看看输入
        // std::cout << mModule << std::endl;
        //for(const auto& input : mModule.get_method("forward").graph()->inputs()) {
        //    std::cout << input->type()->str() << std::endl;  
        //}
        //考虑一下gpu？
    }
    catch (const c10::Error& e){
        std::cerr << "Error load the model:" << e.msg() << std::endl;
    }
    //类名
    std::ifstream ifs(class_path);
    std::string name = "";

    while(getline(ifs,name)) {
        mClassnames.push_back(name);
    }
    //不确定这个要不要变
    mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird", "cat",
                      "dog", "horse", "sheep", "crow", "bear"};

    mMask = cv::Mat::zeros(mRGB.rows, mRGB.cols, CV_8UC1);
    //needUpdateMask = true;
    allDelete = false;
}
YDETRDetector::YDETRDetector() : BaseDetection() {
    //好像是禁用优化啥的
    torch::jit::setTensorExprFuserEnabled(false);
    //加载模型
    try{
        mModule = torch::jit::load("rtdetr-l.torchscript");
        //mModule = torch::jit::load("rtdetr-l.torchscript");
        mModule.eval();
        // 打印一下参数看看输入
        // std::cout << mModule << std::endl;
        //for(const auto& input : mModule.get_method("forward").graph()->inputs()) {
        //    std::cout << input->type()->str() << std::endl;  
        //}
        //考虑一下gpu？
    }
    catch (const c10::Error& e){
        std::cerr << "Error load the model:" << e.msg() << std::endl;
    }
    //类名
    std::ifstream ifs("coco.names");
    std::string name = "";

    while(getline(ifs,name)) {
        mClassnames.push_back(name);
    }
    //不确定这个要不要变
    mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird", "cat",
                      "dog", "horse", "sheep", "crow", "bear"};
    std::cout << "----DETRDetector init----\n";
}
bool YDETRDetector::Detect() {
    std::cout << "DETR detecting\n";
    if(mRGB.empty())
    {
        std::cout << "Read RGB failed!" << std::endl;
        return false;
    }
    //每两帧重置mask；
    //if(needUpdateMask)
    mMask = cv::Mat::zeros(mRGB.rows, mRGB.cols, CV_8UC1);
    //needUpdateMask = !needUpdateMask;
    //预处理图片大小和格式
    cv::Mat img;
    cv::resize(mRGB,img,cv::Size(640,480));
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    torch::Tensor imgTensor = torch::from_blob(img.data,{img.rows, img.cols, 3},torch::kByte);
    imgTensor = imgTensor.permute({2,0,1});
    imgTensor = imgTensor.toType(torch::kFloat);
    imgTensor = imgTensor.div(255);
    //添加batch？DETR需要吗？
    imgTensor = imgTensor.unsqueeze(0);
    //推理得到结果
    try{
        //Tensor类型的IValue
        auto result = mModule.forward({imgTensor});
        //将c10::IValue对象解析
        torch::Tensor result_tensor= result.toTensor();
        // 后处理
        auto dets = PostProcess(result_tensor);
        
        //测试输出det
        if (dets.size() > 0){
            for(size_t i=0; i < dets.size(); ++ i) {
                int cls = dets[i][0].item<int>();
                //这里要缩放到salm系统的合适大小而不是模型输入而且不用归一化了
                float left = dets[i][1].item<float>() * mRGB.cols;
                float top = dets[i][2].item<float>() * mRGB.rows;
                float right = dets[i][3].item<float>() * mRGB.cols;
                float bottom = dets[i][4].item<float>() * mRGB.rows;
                // float score = dets[i][5].item<float>();
                cv::Rect2i DetectArea(left, top, (right - left), (bottom - top));
                mmDetectMap[mClassnames[cls]].push_back(DetectArea);

                if (count(mvDynamicNames.begin(), mvDynamicNames.end(), mClassnames[cls]))
                {
                    //再加一个mask更新
                    
                    cv::Rect2i DynamicArea(left, top, (right - left), (bottom - top));
                    
                    //这一步现在只是框，可以进一步变为mask。。。之后再说，
                    mvDynamicArea.push_back(DynamicArea);
                    maskUpdate(DynamicArea);
                }
            }
        }
        return true;
    }catch (const c10::Error& e) {
        std::cerr << "Error forward the model:" << e.msg() << std::endl;
        return false;
    }
    
    //可视化结果和保存到成员变量方便后面用

}
std::vector<at::Tensor> YDETRDetector::PostProcess(const at::Tensor& preds) {
    //确保在CPU上不然拿不了数据
    float th = 0.4;
    auto inputs_cpu = preds.to(torch::kCPU);
    //std::cout << "inputs_cpu:" << inputs_cpu.index({0,2}) << std::endl;
    //std::cout << "inputs_cpu:" << inputs_cpu.index({0,2}).sizes() << std::endl;

    //获取维度数量信息
    int numQuery = inputs_cpu.size(1);
    int numClass = inputs_cpu.size(2) - 4;

    //先造一个输出向量
    std::vector<torch::Tensor> vTmpOutput;

    //[1,300,84]
    auto scores = preds.slice(2,4,inputs_cpu.size(2));
    auto boxes = preds.slice(2,0,4);
    for(int i = 0; i < numQuery; i++) {
        //如果当前query的最大分数大于阈值就和对应box放入输出向量中
        auto max_scores_class = torch::max(scores,2);

        auto max_conf = std::get<0>(max_scores_class);
        auto max_cls = std::get<1>(max_scores_class);

        float* score_ptr = max_conf.data_ptr<float>();
        int64_t* class_ptr = max_cls.data_ptr<int64_t>();

        if(score_ptr[i] > th) {
            //拿cls
            int64_t class_id = class_ptr[i];
            //std::cout << "class_ptr:" << class_ptr[i] << std::endl;
            //拿box
            auto box = boxes.index({0,i,torch::indexing::Slice()});
            //std::cout << "box:" << box << std::endl;
            //要防止越界
            box[0] = torch::max(box[0] - box[2] / 2, torch::tensor(0.0f));
            box[1] = torch::max(box[1] - box[3] / 2, torch::tensor(0.0f));
            box[2] = torch::min(box[0] + box[2], torch::tensor(1.0f));
            box[3] = torch::min(box[1] + box[3], torch::tensor(1.0f));
            //std::cout << "box:" << box << std::endl;
            //拿分数
            float score = score_ptr[i];
            //std::cout << "score:" << score << std::endl;
            
            //拼接
            std::vector<float> concat_data = {(float)class_id, 
                                            box[0].item<float>(), 
                                            box[1].item<float>(),
                                            box[2].item<float>(), 
                                            box[3].item<float>(), 
                                            score};
                                            
            vTmpOutput.push_back(torch::tensor(concat_data));
        }
    }
    //std::cout << "vTmpOutput:" << vTmpOutput.size() << std::endl;

    
    return vTmpOutput;
}

void YDETRDetector::maskUpdate(cv::Rect2i& DynamicArea) {
    //cv::Mat localMask = cv::Mat::zeros(mRGB.rows, mRGB.cols, CV_8UC1);
    cv::Rect2i box = DynamicArea;
    cv::Mat imDepthLocal = mDepth.clone();
    cv::Mat imBoxTight = imDepthLocal(box);

	std::vector<float> validPixelsInBox;
	for (int y = 0; y < imBoxTight.rows; y++)
	{
		for (int x = 0; x < imBoxTight.cols; x++)
		{
			const float val = imBoxTight.at<float>(y, x);
			if (val > 0) validPixelsInBox.push_back(val);
		}
	}

	std::sort(validPixelsInBox.begin(), validPixelsInBox.end());
	float median = validPixelsInBox[validPixelsInBox.size() / 2];
	float third = validPixelsInBox[validPixelsInBox.size() / 3];
	float value = (median + third) / 2.0;

    std::vector<float> differences;
    differences.reserve(validPixelsInBox.size());
    for(const float& val : validPixelsInBox) {
        //减的是中位数。。
        differences.push_back(std::abs(val - median));
    }
    // 对差值排序
    std::sort(differences.begin(), differences.end());
    // 使用差值分布的分位数作为阈值
    float threshold = differences[differences.size() * 0.68];  // 68%分位数

    //在这里扭曲了。。
    int expansion = 30;
    // 计算扩展后的区域（需要边界检查）
    int x1 = std::max(0, box.x - expansion);
    int y1 = std::max(0, box.y - expansion);
    int x2 = std::min(mRGB.cols, box.x + box.width + expansion);
    int y2 = std::min(mRGB.rows, box.y + box.height + expansion);

    int cnt = 0;
    for (int y = 0; y < imDepthLocal.rows; ++y)
	{
		for (int x = 0; x < imDepthLocal.cols; ++x)
		{
			if (
                imDepthLocal.at<float>(y, x) > 0 && 
                ((imDepthLocal.at<float>(y, x)-median) < threshold)
                ) {
                    mMask.at<uchar>(y, x*2) = 1;
                    mMask.at<uchar>(y, x*2+1) = 1;
                    cnt += 2;
                }
                //我也不知道这里为什么要*2，
                
		}
	}
    if(cnt / DynamicArea.area() < 0.25) {allDelete = true;}
    else {allDelete = false;}
    //cv::imwrite("/mnt/workspace/YOLO_ORB_SLAM3/res/dmask.png", mMask);
}