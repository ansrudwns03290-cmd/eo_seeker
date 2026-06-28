#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class AcquisitionManager {
public:
    AcquisitionManager();

    // [핵심] 1. 초기 표적 설정: 사용자가 마우스로 선택한 이미지 조각에서 특징 추출
    void setTargetModel(const cv::Mat& roiImg);
  
    // 표적 상실 시 후보 탐색
    bool detectCandidateInPredictArea(const cv::Mat& processedGrayImg,
                                        const cv::Rect& predictedRect,
                                        int windowSize,
                                        cv::Rect& outCandidateBox);
    
                                        // 상태 및 정보 관리
    double getTargetRatio() const { return m_targetRatio; }

    bool isFeatureRich() const { return m_isFeatureRich; }
    
    /**
     * @brief 추출된 표적의 ORB 기술자(지문)를 반환
     * @return cv::Mat 타입의 기술자 행렬
     */
    cv::Mat getTargetDescriptors() const { return m_targetDescriptors; }
    cv::Mat getTargetTemplate() const { return m_targetTemplate; }

    bool verifyCandidateWithORB(const cv::Mat& candidateROI);

    void updateTargetModel(const cv::Mat& newTemplate, const cv::Mat& newDescriptors);

private:
    bool findLargestObject(const cv::Mat& binaryImg, cv::Rect& outRect);

    /**
     * @brief 표적 크기에 따라 패딩을 적용한 템플릿 이미지를 생성 (gray 변환 포함)
     * @param sourceImg 템플릿을 잘라낼 원본 이미지
     * @param objectRect 표적의 실제 박스 좌표
     * @return gray 변환된 템플릿 이미지
     */
    cv::Mat buildTemplate(const cv::Mat& sourceImg, const cv::Rect& objectRect);
    cv::Mat buildTemplate(const cv::Mat& croppedImg); // 이미 crop된 이미지에서 gray 변환만 수행 (updateTargetModel용)
    bool m_isFeatureRich = false; // 특징점이 충분히 추출되었는지 여부

    double m_targetRatio = 1.0; // 재획득 시 필터링을 위한 가로세로비

    // ORB 특징점 매칭 관련 (표적의 '몽타주' 데이터)
    cv::Ptr<cv::ORB> m_orb;
    cv::Mat m_targetDescriptors;
    cv::Mat m_targetTemplate;
    std::vector<cv::KeyPoint> m_targetKeypoints;
};