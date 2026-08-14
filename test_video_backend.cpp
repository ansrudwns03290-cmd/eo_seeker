#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
    std::string testDir = "C:/eo_seeker/data"; // 실제 테스트 영상 폴더 경로로 수정

    for (const auto& entry : fs::directory_iterator(testDir)) {
        if (entry.path().extension() != ".mp4") continue;

        cv::VideoCapture cap(entry.path().string(), cv::CAP_MSMF);
        if (!cap.isOpened()) {
            std::cout << "[FAIL] " << entry.path().filename() << " - MSMF로 열기 실패\n";
            continue;
        }

        cv::Mat frame;
        cap >> frame;
        std::cout << (frame.empty()
            ? "[FAIL] " + entry.path().filename().string() + " - 프레임 읽기 실패\n"
            : "[OK]   " + entry.path().filename().string() + " - " 
              + std::to_string(frame.cols) + "x" + std::to_string(frame.rows) + "\n");
    }
    return 0;
}