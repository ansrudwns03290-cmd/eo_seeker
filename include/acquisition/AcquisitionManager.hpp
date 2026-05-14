#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class AcquisitionManager {
public:
    AcquisitionManager();

    // [핵심] 1. 초기 표적 설정: 사용자가 마우스로 선택한 이미지 조각에서 특징 추출
    void setTargetModel(const cv::Mat& roiImg);
  
    // [미래 구현] 2. 표적 재획득: 추적 상실 시 Kalman 예측 위치 근처를 탐색
    // 지금은 선언만 하고, 나중에 Kalman Filter 모듈 완성 후 구현
    bool reAcquire(const cv::Mat& searchArea, cv::Rect predictedRect, cv::Rect& foundTarget);

    // 상태 및 정보 관리
    double getTargetRatio() const { return m_targetRatio; }

    /**
     * @brief 추출된 표적의 ORB 기술자(지문)를 반환
     * @return cv::Mat 타입의 기술자 행렬
     */
    cv::Mat getTargetDescriptors() const { return m_targetDescriptors; }

private:
    bool findLargestObject(const cv::Mat& binaryImg, cv::Rect& outRect);
    
    double m_targetRatio = 1.0; // 재획득 시 필터링을 위한 가로세로비

    // ORB 특징점 매칭 관련 (표적의 '몽타주' 데이터)
    cv::Ptr<cv::ORB> m_orb;
    cv::Mat m_targetDescriptors;
    std::vector<cv::KeyPoint> m_targetKeypoints;
};