#include "YRTDETRDetector.h"

DETRDetector::DETRDetector(const std::string& model_path, const std::string& class_path){
    
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
        //导出模型时代码里有一行transformer有一行把位置编码显式调回了cpu。。。。
        mModule.to(torch::kCUDA);
        
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
}
void DETRDetector::GetImage(cv::Mat& RGB){

    mRGB = RGB;
}
bool DETRDetector::Detect(){
    if(mRGB.empty())
    {
        std::cout << "Read RGB failed!" << std::endl;
        return false;
    }
    //预处理图片大小和格式
    cv::Mat img;
    cv::resize(mRGB,img,cv::Size(640,480));
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    torch::Tensor imgTensor = torch::from_blob(img.data,{img.rows, img.cols, 3},torch::kByte);
    imgTensor = imgTensor.permute({2,0,1});
    imgTensor = imgTensor.toType(torch::kFloat);
    imgTensor = imgTensor.div(255);
    //添加batch？DETR需要吗？
    imgTensor = imgTensor.unsqueeze(0).to(torch::kCUDA);
    //推理得到结果
    try{
        //Tensor类型的IValue
        for (const auto& p : mModule.parameters()) {
                std::cout << "Model parameter device: " << p.device() << std::endl;
            }
        std::cout << "Device: " << imgTensor.device() << std::endl;
        auto result = mModule.forward({imgTensor});
        //将c10::IValue对象解析
        torch::Tensor result_tensor= result.toTensor().to(torch::kCPU);
        std::cout << "Oresult_tensorutput: " << result_tensor.sizes() << std::endl;
        // 后处理
        auto dets = PostProcess(result_tensor,0.35);
        //测试输出det
        //std::cout << "det: " << dets.size() << std::endl;
        //std::cout << "det: " << dets[0].sizes() << std::endl;
        //std::cout << "det: " << dets[0].sizes()[0] << std::endl;
        std::cout << "det[0][0]: " << dets[0][0] << std::endl;
        std::cout << "det[0][0]: " << dets[0][1] << std::endl;
        std::cout << "det[0][0]: " << dets[0][5] << std::endl;
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
                    cv::Rect2i DynamicArea(left, top, (right - left), (bottom - top));
                    //这一步现在只是框，可以进一步变为mask。。。之后再说，
                    mvDynamicArea.push_back(DynamicArea);
                }
            }
            if (mvDynamicArea.size() == 0)
            {
                cv::Rect2i tDynamicArea(1, 1, 1, 1);
                mvDynamicArea.push_back(tDynamicArea);
            }
        }
        return true;
    }catch (const c10::Error& e) {
        std::cerr << "Error forward the model:" << e.msg() << std::endl;
        return false;
    }
    
    //可视化结果和保存到成员变量方便后面用

}
std::vector<torch::Tensor> DETRDetector::PostProcess(const torch::Tensor& inputs, float th){
    //确保在CPU上不然拿不了数据
    auto inputs_cpu = inputs.to(torch::kCPU);
    //std::cout << "inputs_cpu:" << inputs_cpu.index({0,2}) << std::endl;
    //std::cout << "inputs_cpu:" << inputs_cpu.index({0,2}).sizes() << std::endl;

    //获取维度数量信息
    int numQuery = inputs_cpu.size(1);
    int numClass = inputs_cpu.size(2) - 4;

    //先造一个输出向量
    std::vector<torch::Tensor> vTmpOutput;

    //[1,300,84]
    auto scores = inputs.slice(2,4,inputs_cpu.size(2));
    auto boxes = inputs.slice(2,0,4);
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
    std::cout << "vTmpOutput:" << vTmpOutput.size() << std::endl;
    return vTmpOutput;
}
void DETRDetector::ClearArea(){
    mvPersonArea.clear();
}
void DETRDetector::ClearImage(){
    mRGB = 0;
}