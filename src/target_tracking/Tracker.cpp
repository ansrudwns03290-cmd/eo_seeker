#include "target_tracking/Tracker.hpp"
#include <iostream>

Tracker::Tracker() 
    : m_confidence(0.0f), m_isInitialized(false) {
    // 생성자에서는 객체를 할당하지 않고 init 호출 시 할당하는 것이 메모리 관리에 유리합니다.
}

Tracker::~Tracker() {}

bool Tracker::init(const cv::Mat& frame, const cv::Rect& bbox) {
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
        m_lastBbox = outBbox;
        m_confidence = 1.0f; // 필요 시 PSR(Peak to Sidelobe Ratio) 값으로 확장 가능
    } else {
        m_confidence = 0.0f;
        m_isInitialized = false; // 추적 실패 시 초기화 상태 해제 (재획득 유도)
        std::cout << "[Tracker] Target LOST" << std::endl;
    }

    return success;
}