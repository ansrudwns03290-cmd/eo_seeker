#include "acquisition/AcquisitionManager.hpp"
#include "target_tracking/Tracker.hpp"

AcquisitionManager::AcquisitionManager() {
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

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

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
        
        m_targetTemplate = objectOnly.clone(); // 템플릿 매칭용으로 물체 영역 전체 저장
        
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
 * @brief Kalman Filter의 예측 좌표와 동적 윈도우 크기를 기반으로 경략 고속 후보 탐색 수행
 * @param processedGrayImg 전처리 모듈에서 넘어온 흑백 이미지
 * @param predictedRect Kalman Filter가 예측한 표적의 위치와 크기 정보
 * @param windowSize FSM이 결정한 동작 탐색 창 크기
 * @param outCandidateBox [출력 변수] 찾은 최종 후보의 박스 좌표
 * @return 후보 검출 성공 여부
 */
bool AcquisitionManager::detectCandidateInPredictArea(const cv::Mat& processedGrayImg,
                                                        const cv::Rect& predictedRect,
                                                        int windowSize,
                                                        cv::Rect& outCandidateBox) {
    if (processedGrayImg.empty() || predictedRect.width <= 0 || predictedRect.height <= 0) {
        std::cerr << "[DEBUG: Acquisition] Invalid input for candidate detection." << std::endl;
        return false;
    }

    // 1. 칼만 예측 중심점 기준으로 동적 windowSize 크기의 탐색 관심 영역 설정
    cv::Point targetCenter(predictedRect.x + predictedRect.width / 2,
                            predictedRect.y + predictedRect.height / 2);
    
    cv::Rect searchRoi;
    searchRoi.width = predictedRect.width + windowSize;
    searchRoi.height = predictedRect.height + windowSize;
    searchRoi.x = targetCenter.x - searchRoi.width / 2;
    searchRoi.y = targetCenter.y - searchRoi.height / 2;

    // 2. 탐색 창이 화면전체 이미지 경계를 벗어나지 않도록 안전 예외 처리
    cv::Rect imageBounds(0, 0, processedGrayImg.cols, processedGrayImg.rows);
    searchRoi = searchRoi & imageBounds;

    if (searchRoi.width <= 0 || searchRoi.height <= 0)  return false;

    cv::Mat croppedSearchImg = processedGrayImg(searchRoi);
    
    // 템플릿 매칭 수행
    cv::Mat matchResult;
    cv::Mat currentTemplate = m_targetTemplate.empty() ? cv::Mat(30, 30, CV_8UC1, cv::Scalar(0)) : m_targetTemplate; // 템플릿이 없는 경우 작은 검은 이미지로 대체
    std::cout << "[DEBUG] Template Type: " << currentTemplate.type() << " | Search Type: " << croppedSearchImg.type() << std::endl;

    cv::matchTemplate(croppedSearchImg, currentTemplate, matchResult, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal;
    cv::Point maxLoc;
    cv::minMaxLoc(matchResult, &minVal, &maxVal, NULL, &maxLoc);

    // 임계값 검증
    const double MATCH_THRESHOLD = 0.6;
    if (maxVal > MATCH_THRESHOLD) {
        // 매칭된 최적 위치로 후보 박스 정의
        cv::Rect bestCandidate(maxLoc.x, maxLoc.y, currentTemplate.cols, currentTemplate.rows);

        // 정밀 검증 (ORB 특징점 기반)
        // 매칭 위치의 이미지만 crop하여 특징점 검증
        cv::Mat candidateROI = croppedSearchImg(bestCandidate);

        if (verifyCandidateWithORB(candidateROI)) {
            // 전체 좌표계로 변환
            outCandidateBox.x = searchRoi.x + bestCandidate.x;
            outCandidateBox.y = searchRoi.y + bestCandidate.y;
            outCandidateBox.width = bestCandidate.width;
            outCandidateBox.height = bestCandidate.height;
            return true;
        }
    }

    return false; // 해당 구역 내 물체가 전혀 감지되지 않음
}

bool AcquisitionManager::verifyCandidateWithORB(const cv::Mat& candidateROI) {
    if (m_targetDescriptors.empty()) {
        std::cout << "[Acquisition] No ORB descriptors available for verification." << std::endl;
        return false;
    }
    
    // 후보 영역에서 특징점 추출
    std::vector<cv::KeyPoint> candidateKeypoints;
    cv::Mat candidateDescriptors;

    m_orb->detectAndCompute(candidateROI, cv::noArray(), candidateKeypoints, candidateDescriptors);

    if (candidateDescriptors.empty()) {
        std::cout << "[Acquisition] No ORB descriptors found in candidate ROI." << std::endl;
        return false;
    }

    // 매칭
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(m_targetDescriptors, candidateDescriptors, matches);

    // 거리 기반 필터링
    double min_dist = 1000.0;
    for (const auto& m : matches) {
        if (m.distance < min_dist) min_dist = m.distance;
    }

    // 통계적 유의미한 매칭 개수 카운트
    int good_matches = 0;
    for (const auto& m : matches) {
        if(m.distance <= std::max(2.0 * min_dist, 30.0)) {
            good_matches++;
        }
    }

    return (good_matches >= 5);
}

void AcquisitionManager::updateTargetModel(const cv::Mat& newTemplate, const cv::Mat& newDescriptors) {
    if (newTemplate.empty() || newDescriptors.empty()) {
        std::cerr << "[Acquisition] Warning: Attempting to update target model with empty template or descriptors." << std::endl;
        return;
    }
    
    cv::Mat grayTemplate;
    if (newTemplate.channels() == 3) {
        cv::cvtColor(newTemplate, grayTemplate, cv::COLOR_BGR2GRAY);
    } else {
        grayTemplate = newTemplate;
    }

    m_targetTemplate = grayTemplate;
    m_targetDescriptors = newDescriptors.clone();
    std::cout << "[Acquisition] Target model updated with new template and descriptors." << std::endl;
}