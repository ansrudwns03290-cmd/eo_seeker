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

/**
 * @brief 초기 표적의 ORB 기술자 데이터를 전달받아 저장
 *  - AcquisitionManager에서 추출된 기술자 행렬을 받아서 Tracker 내부에
 * @param descriptors AcquisitionManager에서 추출된 ORB 기술자 행렬
 */
void Tracker::setTargetDescriptors(const cv::Mat& descriptors) {
    if (!descriptors.empty()) {
        m_targetDescriptors = descriptors.clone();
        std::cout << "[Tracker] Descriptors received! Size: " << m_targetDescriptors.rows << " points" << std::endl;
    }
}

/**
 * @brief KCF 추적기 초기화
 * @param frame 초기 프레임 이미지
 * @param bbox 초기 표적 영역 (AcquisitionManager에서 전달받은 ROI)
 * @return 초기화 성공 여부
 */
bool Tracker::init(const cv::Mat& frame, const cv::Rect& bbox) {
    if (m_isInitialized) return true;
    
    if (frame.empty() || bbox.width <= 0 || bbox.height <= 0) {
        std::cerr << "[Tracker] Invalid Init Data!" << std::endl;
        return false;
    }

    try {
        cv::Rect safeRoi = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safeRoi.width <= 0 || safeRoi.height <= 0) return false;

        // 원본 정답지 템플릿 저장 (NCC 검증용)
        m_targetTemplate = frame(safeRoi).clone();
        
        // 작은 표적 강제 해상도 업스케일링
        cv::Mat kcfInputFrame = frame.clone();
        cv::Rect kcfInputBbox = bbox;

        // 표적 가로 길이가 60픽셀보다 작다면 업스케일링 적용 (KCF가 작은 물체를 잘 못 잡는 경우 보완)
        if (bbox.width < 60) {
            std::cout << "[Tracker:Upscaling] Target too small (" << bbox.width
                        << "px). Scaling up image by 2x for KCF." << std::endl;

            // 전체 이미지 가로세로 2배 확대
            cv::resize(frame, kcfInputFrame, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);

            // 가상 이미지 해상도에 맞춰 KCF에 입력할 사각형 좌표도 2배 확대
            kcfInputBbox.x = bbox.x * 2;
            kcfInputBbox.y = bbox.y * 2;
            kcfInputBbox.width = bbox.width * 2;
            kcfInputBbox.height = bbox.height * 2;
        }

        // 입력받은 bbox보다 약간 더 넓은 구역을 KCF의 모태 박스로 지정합니다.
        // 가로세로를 약 1.3배 ~ 1.5배 키워서 탐색 범위(윈도우)를 강제로 확장합니다.
        float scale_factor = 1.3f; 
        
        cv::Rect enlarged_bbox;
        enlarged_bbox.width = static_cast<int>(kcfInputBbox.width * scale_factor);
        enlarged_bbox.height = static_cast<int>(kcfInputBbox.height * scale_factor);
        
        // 박스가 커지면서 중심점이 틀어지지 않도록 좌상단(x, y) 좌표를 보정합니다.
        enlarged_bbox.x = kcfInputBbox.x - (enlarged_bbox.width - kcfInputBbox.width) / 2;
        enlarged_bbox.y = kcfInputBbox.y - (enlarged_bbox.height - kcfInputBbox.height) / 2;

        // 화면 밖으로 박스가 나가지 않도록 경계 안전 처리
        cv::Rect img_rect(0, 0, kcfInputFrame.cols, kcfInputFrame.rows);
        enlarged_bbox = enlarged_bbox & img_rect;

        // 기본 구조체로 안전하게 생성
        m_tracker = cv::TrackerKCF::create();
        
        // ★ 확장된 박스로 초기화 수행 (탐색 윈도우가 자동으로 넓어짐)
        m_tracker->init(kcfInputFrame, enlarged_bbox);
        
        m_isInitialized = true;
        m_lastBbox = bbox;
        m_confidence = 1.0f;
        m_frameCount = 0; // 초기화 시 프레임 카운터 리셋

        std::cout << "[Tracker] KCF Initialized successfully." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "[Tracker] Init Exception: " << e.what() << std::endl;
        m_isInitialized = false;
        return false;
    }

    return true;
}

/**
 * @brief 추적기 업데이트
 * @param frame 현재 프레임 이미지
 * @param outBbox 업데이트된 표적 영역
 * @return 추적 성공 여부
 */
bool Tracker::update(const cv::Mat& frame, cv::Rect& outBbox) {
    if (!m_isInitialized || frame.empty()) {
        return false;
    }

    cv::Mat kcfInputFrame = frame;
    cv::Rect virtualBbox;

    // 최초 등록 시 표적이 작아 업스케일링 했는지 검사
    // m_lastBbox의 원본 가로 크기가 60 미만이었다면, 매 프레임 이미지 키우는 기법 적용
    bool isUpscaledMode = (m_lastBbox.width < 60);

    if (isUpscaledMode) {
        // 현재 들어온 640x480 프레임을 2배 확대
        cv::resize(frame, kcfInputFrame, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
    }

    // 3. update 함수는 추적 성공 여부를 bool로 반환합니다.
    bool success = m_tracker->update(kcfInputFrame, virtualBbox);

    if (success) {
        m_frameCount++; // 프레임 카운터 증가

        if (isUpscaledMode) {
            // KCF가 2배 확대된 이미지에서 찾은 좌표를 원래 크기로 보정
            outBbox.x /= 2;
            outBbox.y /= 2;
            outBbox.width /= 2;
            outBbox.height /= 2;
        } else{
            outBbox = virtualBbox; // 원래 크기에서 찾은 좌표 그대로 사용
        }

        // 2. 현재 추적된 영역에서 신뢰도 검증 (ROI 안전 처리 포함)
        cv::Rect safeRoi = outBbox & cv::Rect(0, 0, frame.cols, frame.rows);

        if (safeRoi.width > 0 && safeRoi.height > 0) {
            cv::Mat currentROI = frame(safeRoi);

            if(m_confidence > 0.7f && m_frameCount % 30 == 0) { 
                m_targetTemplate = currentROI.clone(); // 신뢰도가 높을 때마다 템플릿 업데이트 (30프레임마다)
                
                std::vector<cv::KeyPoint> kp;
                m_orb->detectAndCompute(currentROI, cv::noArray(), kp, m_targetDescriptors);
                std::cout << "[Tracker] Template Updated! New Descriptor Count: " << m_targetDescriptors.rows << std::endl;
            }

            m_confidence = verifyTarget(currentROI);
        } else {
            m_confidence = 0.0f; // 안전한 ROI가 없으면 신뢰도 0으로 간주
        }

        m_lastBbox = outBbox; // 마지막 성공한 위치 업데이트
    } else {
        m_confidence = 0.0f;
        m_isInitialized = false; // 추적 실패 시 초기화 상태 해제 (재획득 유도)
        std::cout << "[Tracker] Target LOST" << std::endl;
    }

    return success;
}

/**
 * @brief 현재 추적된 표적의 신뢰도를 검증
 * @param currentROI 현재 추적된 표적 영역
 * @return 신뢰도 점수
 */
float Tracker::verifyTarget(const cv::Mat& currentROI) {
    if (currentROI.empty()) return 0.0f;

    // 1. 특징점 데이터가 있다면 ORB 시도
    const int MIN_DESCRIPTOR_COUNT = 7; 
    
    if (m_targetDescriptors.rows >= MIN_DESCRIPTOR_COUNT) {
        float orbScore = calculateORBConfidence(currentROI);
        
        // ORB 매칭 결과가 유의미하다면 즉시 반환
        if (orbScore > 0.1f) return orbScore; 
    }

    // 2. 특징점이 없거나 ORB 점수가 너무 낮으면 NCC 시도
    if (!m_targetTemplate.empty()) {
        return calculateNCCConfidence(currentROI);
    }

    return 0.2f; // 둘 다 실패 시
}

/**
 * @brief ORB 매칭을 통한 신뢰도 계산
 * @param currentROI 현재 추적된 표적 영역
 * @return ORB 매칭 기반 신뢰도 점수 (0.0 ~ 1.0)
 * - 매칭된 특징점 수를 기반으로 간단히 계산하며, 15개 이상의 매칭은 1.0으로 간주
 * - 매칭 거리가 80 이하인 경우를 좋은 매칭으로 간주 (경험적 기준)
 */
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

    //std::cout << "[Tracker: Debug] ORB Matches: " << goodMatchCount << std::endl;
    return std::min(static_cast<float>(goodMatchCount) / 15.0f, 1.0f);
}

/**
 * @brief NCC 매칭을 통한 신뢰도 계산
 * @param currentROI 현재 추적된 표적 영역
 * @return NCC 매칭 기반 신뢰도 점수 (0.0 ~ 1.0)
 * - NCC 결과는 -1.0 ~ 1.0 범위이므로, 0.0 미만은 0.0으로 보정하여 반환
 */
float Tracker::calculateNCCConfidence(const cv::Mat& currentROI) {
    if (m_targetTemplate.empty() || currentROI.empty()) return 0.0f;

    cv::Mat res, resizedROI;
    cv::resize(currentROI, resizedROI, m_targetTemplate.size());
    
    cv::matchTemplate(resizedROI, m_targetTemplate, res, cv::TM_CCOEFF_NORMED);
    double minVal, maxVal;
    cv::minMaxLoc(res, &minVal, &maxVal);
    
    float rawNcc = static_cast<float>(maxVal);

    // ★ [핵심 보정 알고리즘] ★
    // rawNcc가 0.45 이상이면 거의 완벽히 잡은 상태이므로 0.9 ~ 1.0으로 매핑하고,
    // 0.2 이하로 떨어지면 완전히 놓친 배경 상태로 매핑합니다.
    float finalConf = 0.0f;
    float minThresh = 0.25f;
    float maxThresh = 0.48f;

    if (rawNcc >= maxThresh) {
        finalConf = 1.0f;
    } else if (rawNcc <= minThresh) {
        finalConf = 0.0f;
    } else {
        // 중간 구간 선형 보정 (0.25 ~ 0.48 사이의 점수를 0.0 ~ 1.0으로 확대)
        finalConf = (rawNcc - minThresh) / (maxThresh - minThresh);
    }
    
    // 디버그용 출력으로 실제 원본 점수와 보정 점수를 같이 모니터링합니다.
    // std::cout << "[NCC Debug] Raw: " << rawNcc << " -> Enhanced Conf: " << finalConf << std::endl;
    
    return finalConf;
}