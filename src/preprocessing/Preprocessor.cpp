#include "preprocessing/Preprocessor.hpp"

Preprocessor::Preprocessor() : m_roi(0, 0, 0, 0) {}
Preprocessor::~Preprocessor() {}

void Preprocessor::setROI(cv::Rect roi) {
    m_roi = roi;
    m_isRoiSet = true;
}

/**
 * @brief 입력 이미지를 흑백 이미지로 변환
 */
bool Preprocessor::process(const Frame& inputFrame, cv::Mat& outputImage) {
    if (inputFrame.image.empty()) return false;

    cv::Mat gray;
    cv::cvtColor(inputFrame.image, gray, cv::COLOR_BGR2GRAY);

    outputImage = gray; // ROI가 없으면 전체 흑백 이미지 반환
    return true;
}