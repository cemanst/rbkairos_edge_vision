#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <vision_msgs/Detection2DArray.h> // NOVO
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include <fstream>
#include <vector>
#include <chrono>

using namespace nvinfer1;

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
    ros::Publisher det_pub; // NOVO: Megafon
    const int INPUT_H = 640;
    const int INPUT_W = 640;

public:
    YoloDetector(ros::NodeHandle& nh, const std::string& engine_path) {
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
        outputIndex = engine->getBindingIndex("output0");

        cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float));
        cudaMalloc(&buffers[outputIndex], 1 * 25200 * 85 * sizeof(float));
        cudaStreamCreate(&stream);

        // Inicijalizacija publisher-a
        det_pub = nh.advertise<vision_msgs::Detection2DArray>("fruit_detections", 1);
        ROS_INFO("YoloDetector: Ready to publish detections.");
    }

    void process(cv::Mat& img, const std::string& frame_id, const ros::Time& stamp) {
        auto start = std::chrono::high_resolution_clock::now();

        // 1. Preprocessing (Resize/Convert/Normalizacija)
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(640, 640));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        // 2. HWC to CHW Copy
        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);
        float* gpu_input = (float*)buffers[inputIndex];
        for (int i = 0; i < 3; ++i) {
            cudaMemcpyAsync(gpu_input + i * 640 * 640, channels[i].ptr<float>(), 
                            640 * 640 * sizeof(float), cudaMemcpyHostToDevice, stream);
        }

        // 3. Inference
        context->enqueueV2(buffers, stream, nullptr);

        // 4. POST-PROCESSING (Povlačenje rezultata na CPU)
        float* cpu_output = new float[25200 * 85];
        cudaMemcpyAsync(cpu_output, buffers[outputIndex], 25200 * 85 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        for (int i = 0; i < 25200; ++i) {
            float obj_conf = cpu_output[i * 85 + 4];
            if (obj_conf > 0.35) {
                float* scores = cpu_output + i * 85 + 5;
                cv::Mat scores_mat(1, 80, CV_32FC1, scores);
                double max_class_score;
                cv::Point class_id;
                cv::minMaxLoc(scores_mat, 0, &max_class_score, 0, &class_id);

                if (max_class_score > 0.25) {
                    float cx = cpu_output[i * 85 + 0];
                    float cy = cpu_output[i * 85 + 1]*0.75; //skalirano za razvucenu sliku 480/640!
                    float w = cpu_output[i * 85 + 2];
                    float h = cpu_output[i * 85 + 3]*0.75;
                    boxes.push_back(cv::Rect(cx - w/2, cy - h/2, w, h));
                    confidences.push_back(obj_conf * max_class_score);
                    classIds.push_back(class_id.x);
                }
            }
        }

        // 5. NMS (Uklanjanje preklapanja)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.35, 0.45, indices);

        // 6. ROS Publish
        vision_msgs::Detection2DArray det_msg;
        det_msg.header.stamp = stamp; // Bitno za merenje latence!
        det_msg.header.frame_id = frame_id;

        for (int idx : indices) {
            vision_msgs::Detection2D d;

            // 1. Puni koordinate
            d.bbox.center.x = boxes[idx].x + boxes[idx].width / 2;
            d.bbox.center.y = boxes[idx].y + boxes[idx].height / 2;
            d.bbox.size_x = boxes[idx].width;
            d.bbox.size_y = boxes[idx].height;

            // 2. Puni hipotezu (ID klase i Score)
            vision_msgs::ObjectHypothesisWithPose hyp;
            hyp.id = classIds[idx];
            hyp.score = confidences[idx];

            // 3. Ubaci hipotezu u detekciju
            d.results.push_back(hyp);

            // 4. TEK SAD ubaci kompletnu detekciju u niz poruke
            det_msg.detections.push_back(d);
        }
        det_pub.publish(det_msg);

        delete[] cpu_output;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        ROS_INFO("Processed frame in %.2f ms", diff.count());
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
        det->process(frame, msg->header.frame_id, msg->header.stamp);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("CV Bridge error: %s", e.what());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "edge_vision_node");
    ros::NodeHandle nh;
    YoloDetector detector(nh, "/home/sajam/catkin_ws/src/jetson_yolo/model/yolov5n.engine");

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 1, 
        boost::bind(callback, _1, &detector), ros::VoidPtr(), image_transport::TransportHints("compressed"));

    ros::spin();
    return 0;
}
