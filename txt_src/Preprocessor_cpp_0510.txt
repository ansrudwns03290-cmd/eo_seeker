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

    // if (m_isRoiSet) {
    //     // ROI 영역이 전체 이미지 크기를 벗어나는지 마지막으로 체크
    //     cv::Rect safeRoi = m_roi & cv::Rect(0, 0, gray.cols, gray.rows);
        
    //     if (safeRoi.width > 0 && safeRoi.height > 0) {
    //         // 이 부분에서 예외가 자주 발생하므로 clone()을 사용하여 안정성 확보
    //         outputImage = gray(safeRoi).clone();
    //         return true;
    //     }
    // }

    outputImage = gray; // ROI가 없으면 전체 흑백 이미지 반환
    return true;
}