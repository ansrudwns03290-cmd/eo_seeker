#include "target_tracking/Tracker.hpp"
#include <iostream>

Tracker::Tracker() 
    : m_confidence(0.0f), m_isInitialized(false) {
    // 생성자에서는 객체를 할당하지 않고 init 호출 시 할당하는 것이 메모리 관리에 유리합니다.
    m_orb = cv::ORB::create(500);
    // ORB는 Binary 기술자이므로 Hamming 거리를 사용하는 매칭기를 생성합니다.
    m_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

Tracker::~Tracker() {}

void Tracker::setTargetDescriptors(const cv::Mat& descriptors) {
    if (!descriptors.empty()) {
        m_targetDescriptors = descriptors.clone();
        std::cout << "[Tracker] Descriptors received! Size: " << m_targetDescriptors.rows << " points" << std::endl;
    }
}

bool Tracker::init(const cv::Mat& frame, const cv::Rect& bbox) {
    if (m_isInitialized) return true;

    if (frame.empty() || bbox.width <= 0 || bbox.height <= 0) {
        std::cerr << "[Tracker] Invalid Init Data!" << std::endl;
        return false;
    }

    try {
        // 1. 최신 OpenCV (4.5.1+) 방식: 알고리즘별 전용 create() 호출
        // opencv_contrib 모듈이 정상 빌드되었다면 아래 코드가 작동합니다.
        m_tracker = cv::TrackerKCF::create();

        // 2. 추적기 초기화 
        // 최신 버전의 init()은 성공 시 void를 반환하므로 try-catch로 성공을 보장합니다.
        m_tracker->init(frame, bbox);
        
        m_isInitialized = true;
        m_lastBbox = bbox;
        m_confidence = 1.0f;
        
        std::cout << "[Tracker] KCF Initialized successfully." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "[Tracker] Init Exception: " << e.what() << std::endl;
        m_isInitialized = false;
        return false;
    }

    return true;
}

bool Tracker::update(const cv::Mat& frame, cv::Rect& outBbox) {
    if (!m_isInitialized || frame.empty()) {
        return false;
    }

    // 3. update 함수는 추적 성공 여부를 bool로 반환합니다.
    bool success = m_tracker->update(frame, outBbox);

    if (success) {
        // 2. 현재 추적된 영역에서 신뢰도 검증 (ROI 안전 처리 포함)
        cv::Rect safeRoi = outBbox & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safeRoi.width > 0 && safeRoi.height > 0) {
            m_confidence = verifyTarget(frame(safeRoi));
        }

        // 3. 신뢰도가 너무 낮으면(예: 0.1 미만) 추적 실패로 간주할 수도 있음
        //if (m_confidence < 0.05f) success = false;
        
        m_lastBbox = outBbox;
    } else {
        m_confidence = 0.0f;
        m_isInitialized = false; // 추적 실패 시 초기화 상태 해제 (재획득 유도)
        std::cout << "[Tracker] Target LOST" << std::endl;
    }

    return success;
}

float Tracker::verifyTarget(const cv::Mat& currentROI) {
    if (currentROI.empty()) return 0.0f;

    // 1. 특징점 데이터가 있다면 ORB 시도
    if (!m_targetDescriptors.empty()) {
        float orbScore = calculateORBConfidence(currentROI);
        // 만약 ORB 매칭이 어느 정도 나온다면 바로 반환
        if (orbScore > 0.1f) return orbScore; 
    }

    // 2. 특징점이 없거나 ORB 점수가 너무 낮으면 NCC 시도
    if (!m_targetTemplate.empty()) {
        return calculateNCCConfidence(currentROI);
    }

    return 0.2f; // 둘 다 실패 시
}

float Tracker::calculateORBConfidence(const cv::Mat& currentROI) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    m_orb->detectAndCompute(currentROI, cv::noArray(), keypoints, descriptors);

    if (descriptors.empty()) return 0.0f;

    std::vector<cv::DMatch> matches;
    m_matcher->match(m_targetDescriptors, descriptors, matches);

    int goodMatchCount = 0;
    for (const auto& match : matches) {
        if (match.distance < 80.0) goodMatchCount++;
    }

    std::cout << "[Tracker: Debug] ORB Matches: " << goodMatchCount << std::endl;
    return std::min(static_cast<float>(goodMatchCount) / 10.0f, 1.0f);
}

float Tracker::calculateNCCConfidence(const cv::Mat& currentROI) {
    cv::Mat res, resizedROI;
    cv::resize(currentROI, resizedROI, m_targetTemplate.size());
    
    cv::matchTemplate(resizedROI, m_targetTemplate, res, cv::TM_CCOEFF_NORMED);
    double minVal, maxVal;
    cv::minMaxLoc(res, &minVal, &maxVal);
    
    std::cout << "[Tracker: Debug] NCC Similarity: " << maxVal << std::endl;
    
    // NCC 결과는 음수가 나올 수 있으므로 0으로 보정
    return std::max(0.0f, static_cast<float>(maxVal));
}