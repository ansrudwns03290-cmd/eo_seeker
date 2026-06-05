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

    // 3. 전체 흑백 프레임에서 탐색 관심 영역만 crop
    // 복사를 하지 않고 참조 매트릭스를 사용하여 메모리 소모 최소화
    cv::Mat croppedSearchImg = processedGrayImg(searchRoi);

    // 4. 잘라낸 영역에서 이진화 수행
    cv::Mat binaryImg;
    cv::threshold(croppedSearchImg, binaryImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binaryImg, binaryImg, cv::MORPH_OPEN, kernel);

    cv::Mat closing_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(binaryImg, binaryImg, cv::MORPH_CLOSE, closing_kernel);

    // 5. 이진화된 탐색 영역에서 가장 큰 물체의 bbox 탐색
    cv::Rect localCandidateBox;
    static int cnt = 0;
    if (findLargestObject(binaryImg, localCandidateBox)) {
        // 6. crop한 국소 좌표계로 나온 후보 박스를 원래 전체 화면 좌표계로 변환
        outCandidateBox.x = searchRoi.x + localCandidateBox.x;
        outCandidateBox.y = searchRoi.y + localCandidateBox.y;
        outCandidateBox.width = localCandidateBox.width;
        outCandidateBox.height = localCandidateBox.height;
        cv::imwrite("C:/eo_seeker/debug_images/detected_candidate" + std::to_string(cnt++) + ".png", binaryImg); // 디버그용 후보 이미지 저장
        std::cout << "[Acquisition: Debug] Detected Candidate Box: " << outCandidateBox << std::endl;
                
        // 7. 가로세로비(Ratio) 3차 검증 검사
        // 모양새가 기존에 등록해둔 전투기 형태와 너무 다르면 가짜 노이즈로 보고 즉시 필터링
        // double currentRatio = static_cast<double>(outCandidateBox.width) / outCandidateBox.height;
        // double ratioError = std::abs(currentRatio - m_targetRatio);
        
        // if (ratioError > 0.8) { // 형상 오차 허용 임계값 (상황에 맞게 조율 가능)
        //     std::cout << "[Acquisition: LOST] 후보를 찾았으나 형상비 규격 미달로 기각 (오차: " << ratioError << ")" << std::endl;
        //     return false;
        // }

        return true; // 노이즈 관문을 모두 뚫고 올라온 최종 단 하나의 '진짜 심증 후보' 확정
    }

    return false; // 해당 구역 내 물체가 전혀 감지되지 않음
}