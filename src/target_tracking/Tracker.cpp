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

        // 원본 정답지 템플릿 저장 (NCC 검증용) - 원본 프레임(1채널 흑백)에서 그대로 복제
        m_targetTemplate = frame(safeRoi).clone();
        
        // 1. 메인에서 들어온 프레임을 조작하기 위해 workingFrame 변수로 먼저 안전 복제
        cv::Mat workingFrame = frame.clone();
        cv::Rect kcfInputBbox = bbox;

        // 표적 가로 길이가 60픽셀보다 작다면 업스케일링 적용
        bool isUpscaledMode = (bbox.width < 60);
        if (isUpscaledMode) {
            std::cout << "[Tracker:Upscaling] Target too small (" << bbox.width
                      << "px). Scaling up image by 2x for KCF." << std::endl;

            // [오타 교정] 복제본인 workingFrame을 소스로 삼아 2배 고속 확대 수행
            cv::resize(workingFrame, workingFrame, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);

            // KCF에 입력할 사각형 좌표도 정확하게 2배 확대
            kcfInputBbox.x = bbox.x * 2;
            kcfInputBbox.y = bbox.y * 2;
            kcfInputBbox.width = bbox.width * 2;
            kcfInputBbox.height = bbox.height * 2;
        }

        // 2. 입력받은 bbox보다 약 1.3배 넓은 구역을 KCF의 모태 박스(탐색 창)로 확장
        float scale_factor = 1.3f; 
        cv::Rect enlarged_bbox;
        enlarged_bbox.width = static_cast<int>(kcfInputBbox.width * scale_factor);
        enlarged_bbox.height = static_cast<int>(kcfInputBbox.height * scale_factor);
        
        // 중심점이 틀어지지 않도록 좌상단(x, y) 좌표 보정
        enlarged_bbox.x = kcfInputBbox.x - (enlarged_bbox.width - kcfInputBbox.width) / 2;
        enlarged_bbox.y = kcfInputBbox.y - (enlarged_bbox.height - kcfInputBbox.height) / 2;

        // 화면 밖으로 박스가 나가지 않도록 경계 안전 처리 (확대된 workingFrame 기준 크기 적용)
        cv::Rect img_rect(0, 0, workingFrame.cols, workingFrame.rows);
        enlarged_bbox = enlarged_bbox & img_rect;

        // 3. [핵심 안전장치]: 크기 변환이 끝난 workingFrame이 1채널 흑백이라면,
        // KCF 코어 내부의 고정 채널 충돌을 방지하기 위해 여기서 최종 3채널 컬러 포맷으로 가공합니다.
        cv::Mat kcfInputFrame;
        if (workingFrame.channels() == 1) {
            cv::cvtColor(workingFrame, kcfInputFrame, cv::COLOR_GRAY2BGR);
        } else {
            kcfInputFrame = workingFrame;
        }

        // KCF 트래커 엔진 인스턴스 생성
        m_tracker = cv::TrackerKCF::create();
        
        // ★ 완벽히 정합된 3채널 이미지와 확장된 박스로 KCF 심장 시동!
        m_tracker->init(kcfInputFrame, enlarged_bbox);
        
        m_isInitialized = true;
        
        // [싱크 교정]: update 함수와의 동역학 매칭을 위해 
        // m_lastBbox에는 업스케일링 여부와 상관없이 무조건 '원본 크기(bbox)'를 저장합니다.
        m_lastBbox = bbox; 
        
        m_confidence = 1.0f;
        m_frameCount = 0; 

        std::cout << "[Tracker] KCF Initialized successfully. (Grayscale Target Ready)" << std::endl;
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

    cv::Mat workingFrame = frame.clone();
    cv::Rect virtualBbox;

    // 최초 등록 시 표적이 작아 업스케일링 했는지 검사
    // m_lastBbox의 원본 가로 크기가 60 미만이었다면, 매 프레임 이미지 키우는 기법 적용
    bool isUpscaledMode = (m_lastBbox.width < 60);

    if (isUpscaledMode) {
        // 현재 들어온 640x480 프레임을 2배 확대
        cv::resize(workingFrame, workingFrame, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
        
        virtualBbox = m_lastBbox;
        virtualBbox.x *= 2;
        virtualBbox.y *= 2;
        virtualBbox.width *= 2;
        virtualBbox.height *= 2;
    } else {
        virtualBbox = m_lastBbox;
    }
    
    // [핵심 안전장치]: 크기 변환(resize)이 완전히 끝난 1채널 최종 데이터를 
    // KCF 트래커 규격에 맞춰 3채널 포맷으로 복제 래핑합니다.
    cv::Mat kcfInputFrame;
    if (workingFrame.channels() == 1) {
        cv::cvtColor(workingFrame, kcfInputFrame, cv::COLOR_GRAY2BGR);
    } else {
        kcfInputFrame = workingFrame;
    }

    // 3. update 함수는 추적 성공 여부를 bool로 반환합니다.
    bool success = m_tracker->update(kcfInputFrame, virtualBbox);

    if (success) {
        m_frameCount++; // 프레임 카운터 증가

        if (isUpscaledMode) {
            // KCF가 2배 확대된 이미지에서 찾은 좌표를 원래 크기로 보정
            outBbox.x = virtualBbox.x / 2;
            outBbox.y = virtualBbox.y / 2;
            outBbox.width = virtualBbox.width / 2;
            outBbox.height = virtualBbox.height / 2;
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
    
    // NCC 매칭을 위해 입력 이미지와 템플릿의 채널 수가 다르면, 안전하게 맞춰주는 전처리 단계
    if (resizedROI.channels() != m_targetTemplate.channels()) {
        if (m_targetTemplate.channels() == 1 && resizedROI.channels() == 3) {
            // 정답지가 흑백인데 입력이 컬러라면, 입력을 흑백으로 변환
            cv::cvtColor(resizedROI, resizedROI, cv::COLOR_BGR2GRAY);
        } 
        else if (m_targetTemplate.channels() == 3 && resizedROI.channels() == 1) {
            // 정답지가 컬러인데 입력이 흑백이라면, 입력을 가짜 컬러로 확장
            cv::cvtColor(resizedROI, resizedROI, cv::COLOR_GRAY2BGR);
        }
    }

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

float Tracker::verifyCandidate(const cv::Mat& currentROI) {
    if (currentROI.empty()) return 0.0f;

    float finalScore = 0.0f;
    bool orbValid = false;

    // 1. 원본 특징점 descriptors가 충분히 존재할 때만 엄격히 매칭
    const int MIN_DESCRIPTOR_COUNT = 7;
    if (m_targetDescriptors.rows >= MIN_DESCRIPTOR_COUNT) {
        float orbScore = calculateORBConfidence(currentROI);

        // REACQUIRE 검증에서는 0.1을 넘겼다고 바로 통과시키지 않고 점수 보관
        if (orbScore >= 0.4f) {
            finalScore = orbScore;
            orbValid = true;
        }
    }

    std::cout << "DEBUG 1: ORB 검증 점수 = " << finalScore << " (Valid: " << orbValid << ")" << std::endl;
    // 2. [하이브리드 NCC 검증]: ORB 점수가 애매하거나 특징점이 부족할 때, 템플릿 픽셀 매칭 점수와 '평균' 계산
    if (!m_targetTemplate.empty()) {
        std::cout << "DEBUG 2: ORB 검증 후 NCC 시도" << std::endl;
        float nccScore = calculateNCCConfidence(currentROI);
        std::cout << "DEBUG 3: NCC 검증 점수 = " << nccScore << std::endl;
        if (orbValid) {
            // ORB도 합격이고 NCC도 합격이면 두 신뢰도의 평균을 내어 정밀도를 극대화 (방산 Seeker 표준)
            // finalScore = (finalScore + nccScore) / 2.0f;
            finalScore = std::max(finalScore, nccScore); // ORB와 NCC 중 더 높은 점수를 최종 신뢰도로 채택하는 보수적 전략
        } else {
            // ORB가 실패했다면 NCC 점수에 전적으로 의존하되, 가산점 없이 정직한 점수 부여
            finalScore = nccScore;
        }
    }

    std::cout << "DEBUG 4: ORB & NCC 검증 점수 = " << finalScore << std::endl;
    // 3. [안전장치]: 매칭에 완전히 실패했다면 기존처럼 0.2점을 주는 관용을 베풀지 않고 '0.0점'으로 칼같이 과락 처리
    if (!orbValid && finalScore < 0.3f) {
        return 0.0f; 
    }

    return finalScore;
}