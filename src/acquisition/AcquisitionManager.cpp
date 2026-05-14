#include "acquisition/AcquisitionManager.hpp"

AcquisitionManager::AcquisitionManager(){
    // ORB 특징점 추출기 초기화
    m_orb = cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 10); 
}
/**
 * @brief 초기 설정한 ROI에서 표적의 ORB 및 가로세로비 저장
 */
void AcquisitionManager::setTargetModel(const cv::Mat& roiImg) {
    if (roiImg.empty()) return;

    // 1. 박스 내 물체 분리를 위한 전처리
    cv::Mat gray, binary;
    if (roiImg.channels() == 3)
        cv::cvtColor(roiImg, gray, cv::COLOR_BGR2GRAY);
    else
        gray = roiImg;

    if (gray.cols < 31 || gray.rows < 31) {
        std::cerr << "[Acquisition] Warning: ROI too small for ORB!" << std::endl;
    }
    
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 2. 물체 윤곽선 추출 및 정밀 모델링
    cv::Rect actualObjectRect;
    if (findLargestObject(binary, actualObjectRect)) {
        // 너무 작게 잘리지 않도록 padding 추가
        int padding;
        if (actualObjectRect.width < 40 || actualObjectRect.height < 40) {
            padding = 10; // 작은 물체는 특징점 확보를 위해 패딩을 넉넉히
        } else {
            padding = 5;  // 큰 물체는 정밀도를 위해 패딩을 작게
        }
        cv::Rect expandedRect = actualObjectRect;
        expandedRect.x = std::max(0, actualObjectRect.x - padding);
        expandedRect.y = std::max(0, actualObjectRect.y - padding);
        expandedRect.width = std::min(roiImg.cols - expandedRect.x, actualObjectRect.width + padding * 2);
        expandedRect.height = std::min(roiImg.rows - expandedRect.y, actualObjectRect.height + padding * 2);

        cv::Mat objectOnly = roiImg(actualObjectRect); // roiImg에서 actualObjectRect 위치의 이미지만 반환
        //cv::Mat objectOnly = roiImg(expandedRect); // 확장된 영역으로 ORB 추출

        m_orb->detectAndCompute(objectOnly, cv::noArray(), m_targetKeypoints, m_targetDescriptors);
        m_targetRatio = static_cast<double>(actualObjectRect.width) / actualObjectRect.height;

        m_isFeatureRich = (m_targetKeypoints.size() >= 10);

        std::cout << "[Acquisition] Mode: " << (m_isFeatureRich ? "ORB-Rich" : "Template-Only") << std::endl;
    } else {
        // 물체 분리 실패 시 박스 전체 사용
        m_orb->detectAndCompute(roiImg, cv::noArray(), m_targetKeypoints, m_targetDescriptors);
        m_targetRatio = static_cast<double>(roiImg.cols) / roiImg.rows;
        std::cout << "[Acquisition] Target Registered (Box Mode)" << std::endl;
    }

    m_orb->detectAndCompute(roiImg(actualObjectRect), cv::noArray(), m_targetKeypoints, m_targetDescriptors);
    std::cout << "[Acquisition: Debug] Extracted ORB Keypoints: " << m_targetKeypoints.size() << std::endl;
}

/**
 * @brief 이진 이미지에서 가장 큰 물체의 bbox를 탐색
 * @param outRect cv::Rect 자료형으로서, 표적이 존재하는 bbox의 위치와 크기 나타냄
 */
bool AcquisitionManager::findLargestObject(const cv::Mat& binaryImg, cv::Rect& outRect) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea = -1.0;
    int bestIdx = -1;

    for (int i = 0; i < (int)contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < 100) continue; 

        if (area > maxArea) {
            maxArea = area;
            bestIdx = i;
        }
    }

    if (bestIdx != -1) {
        outRect = cv::boundingRect(contours[bestIdx]);
        return true;
    }
    return false;
}

/**
 * @brief [미래 구현] Kalman Filter의 예측 좌표를 기반으로 표적 재탐색
 */
bool AcquisitionManager::reAcquire(const cv::Mat& searchArea, cv::Rect predictedRect, cv::Rect& foundTarget) {
    // TODO: Phase 3에서 구현 예정
    // 1. predictedRect 주변으로 searchArea를 제한 (ROI 설정)
    // 2. 해당 영역에서 findContours 또는 ORB 매칭 수행
    // 3. m_targetDescriptors와 비교하여 유사도 검증
    
    return false; // 현재 단계에서는 항상 false 반환 (미구현)
}