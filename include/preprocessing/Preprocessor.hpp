#pragma once
#include <opencv2/opencv.hpp>
#include "common/Frame.hpp" // 업로드해주신 Frame 구조체 포함 [cite: 2750]

class Preprocessor {
public:
    Preprocessor();
    ~Preprocessor();

    // 핵심 전처리 함수: Frame을 입력받아 흑백 변환 및 ROI Crop 수행 [cite: 2127, 2140]
    bool process(const Frame& inputFrame, cv::Mat& outputImage);

    // ROI(관심 영역) 설정 함수: 표적 획득 모듈로부터 좌표를 전달받음 [cite: 2137, 2143]
    void setROI(cv::Rect roi);

private:
    /*m은 member의 약자로서, 클래스가 지속적으로 가지고 있는 멤버 변수임을 타나냄*/
    cv::Rect m_roi;           // 현재 설정된 관심 영역 [cite: 2145]
    bool m_isRoiSet = false;  // ROI 설정 여부
};