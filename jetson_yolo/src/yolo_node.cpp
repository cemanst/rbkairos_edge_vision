#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include <fstream>
#include <vector>

using namespace nvinfer1;

// Minimalistički logger za TensorRT
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
} gLogger;

class YoloDetector {
private:
    ICudaEngine* engine;
    IExecutionContext* context;
    void* buffers[2];
    cudaStream_t stream;
    int inputIndex, outputIndex;
    const int INPUT_H = 640;
    const int INPUT_W = 640;

public:
    YoloDetector(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        char* data = new char[size];
        file.read(data, size);

        IRuntime* runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(data, size);
        context = engine->createExecutionContext();
        delete[] data;

        inputIndex = engine->getBindingIndex("images");
        outputIndex = engine->getBindingIndex("output");

        cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float));
        cudaMalloc(&buffers[outputIndex], 1 * 25200 * 85 * sizeof(float)); // YOLOv5 output size
        cudaStreamCreate(&stream);
    }

    void process(cv::Mat& img) {
        // Pre-processing
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        // HWC to CHW (TensorRT format)
        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);
        float* gpu_input = (float*)buffers[inputIndex];
        for (int i = 0; i < 3; ++i) {
            cudaMemcpyAsync(gpu_input + i * INPUT_H * INPUT_W, channels[i].ptr<float>(), 
                            INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);
        }

        // Inference
        context->enqueueV2(buffers, stream, nullptr);
        
        // Ovde bi išao Post-processing i slanje koordinata Franka ruci
        ROS_INFO("Fruit detection inference successful on Edge.");
    }

    ~YoloDetector() {
        cudaStreamDestroy(stream);
        cudaFree(buffers[inputIndex]);
        cudaFree(buffers[outputIndex]);
        context->destroy();
        engine->destroy();
    }
};

void callback(const sensor_msgs::ImageConstPtr& msg, YoloDetector* det) {
    try {
        cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
        det->process(frame);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge error: %s", e.what());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "edge_vision_node");
    ros::NodeHandle nh;
	//model path 
    YoloDetector detector("/home/sajam/catkin_ws/src/jetson_yolo/model/yolov5n.engine");

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 1, 
        boost::bind(callback, _1, &detector), ros::VoidPtr(), image_transport::TransportHints("compressed"));

    ros::spin();
    return 0;
}

