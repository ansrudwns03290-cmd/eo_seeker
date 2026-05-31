#include "target_tracking/Tracker.hpp"
#include <iostream>
#include <algorithm>

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

            cv::resize(workingFrame, workingFrame, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);

            // KCF에 입력할 사각형 좌표도 정확하게 2배 확대
            kcfInputBbox.x = bbox.x * 2;
            kcfInputBbox.y = bbox.y * 2;
            kcfInputBbox.width = bbox.width * 2;
            kcfInputBbox.height = bbox.height * 2;
        }

        cv::Rect img_rect(0, 0, workingFrame.cols, workingFrame.rows);
        kcfInputBbox = kcfInputBbox & img_rect;

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
        m_tracker->init(kcfInputFrame, kcfInputBbox);
        
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
        cv::Rect safeRoi = virtualBbox & cv::Rect(0, 0, frame.cols, frame.rows);
        
        if (safeRoi.width > 0 && safeRoi.height > 0) {
            cv::Mat currentROI = kcfInputFrame(safeRoi).clone();

            if(m_confidence > 0.7f && m_frameCount % 30 == 0) { 
                // ====================================================================
                // 🌟 [메모리 교착 해결 방어선]: 안전한 1채널 흑백 독립 복제본 생성
                // ====================================================================
                cv::Mat grayROI;
                
                // 1. 만약 currentROI가 3채널 컬러라면, 안전하게 1채널 흑백으로 강제 변환합니다.
                if (currentROI.channels() == 3) {
                    cv::cvtColor(currentROI, grayROI, cv::COLOR_BGR2GRAY);
                } else {
                    // 이미 1채널이라면, 주소 링크를 끊기 위해 반드시 깊은 복사(.clone())를 수행합니다.
                    grayROI = currentROI.clone(); 
                }

                // 2. 완벽하게 격리된 1채널 흑백 이미지(grayROI)를 주입하므로 절대 멈추지 않습니다.
                cv::Mat binImg;
                cv::threshold(grayROI, binImg, 70, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);               
                
                // 모폴로지 연산으로 잔먼지 제거
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                cv::morphologyEx(binImg, binImg, cv::MORPH_OPEN, kernel);

                // 윤곽선 추출
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(binImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                int maxObjectWidth = 0;
                for (const auto& contour : contours) {
                    cv::Rect objectBox = cv::boundingRect(contour);
                    if (objectBox.width > maxObjectWidth && objectBox.width > 10) {
                        maxObjectWidth = objectBox.width;
                    }
                }

                
                int currentOriginalWidth = isUpscaledMode ? (maxObjectWidth / 2) : maxObjectWidth; // 원본 크기로 환산
                    
                // 잡음 때문에 실측 실패하면 안전장치로 outBbox라도 백업
                if (currentOriginalWidth <= 0) {
                    currentOriginalWidth = outBbox.width;
                }

                std::cout << "[Tracker] KCF 크기: " << outBbox.width
                            << "px, | 이진화 실측 크기" << currentOriginalWidth << "px" << std::endl;

                // 표적의 실제 크기를 기반으로 추적기 스위칭 판단
                if (isUpscaledMode && currentOriginalWidth >= 75) {
                    std::cout << "[Tracker] 표적 크기 75px 도달! 원본 모드로 전환" << std::endl;

                    cv::Rect origSafeRoi = outBbox & cv::Rect(0, 0, frame.cols, frame.rows);
                    m_targetTemplate = frame(origSafeRoi).clone();

                    // 업스케일링 모드에서 원본 모드로 전환 시, ORB 기술자도 원본 크기에 맞게 재추출하여 업데이트합니다.
                    std::vector<cv::KeyPoint> kp;
                    m_targetDescriptors.release(); // 기존 기술자 데이터 해제
                    m_orb->detectAndCompute(m_targetTemplate, cv::noArray(), kp, m_targetDescriptors);

                    // KCF 트래커도 원본 크기에 맞춰 재부팅합니다.
                    m_tracker.release(); // 기존 트래커 객체 해제
                    m_tracker = cv::TrackerKCF::create();

                    cv::Mat rawInitFrame;
                    if (frame.channels() == 1) cv::cvtColor(frame, rawInitFrame, cv::COLOR_GRAY2BGR);
                    else rawInitFrame = frame;

                    m_tracker->init(rawInitFrame, origSafeRoi); // 원본 프레임과 원본 크기의 박스로 재부팅
                    m_lastBbox = origSafeRoi; // 재부팅 완료 후 초기화 상태 확립
                }
                else if (!isUpscaledMode && currentOriginalWidth < 50) {
                    std::cout << "[Tracker] 표적 크기 50px 미만! 업스케일 모드로 전환" << std::endl;

                    cv::Rect origSafeRoi = outBbox & cv::Rect(0, 0, frame.cols, frame.rows);
                    m_targetTemplate = frame(origSafeRoi).clone();

                    std::vector<cv::KeyPoint> kp;
                    m_targetDescriptors.release();
                    m_orb->detectAndCompute(m_targetTemplate, cv::noArray(), kp, m_targetDescriptors);

                    m_tracker.release();
                    m_tracker = cv::TrackerKCF::create();
                    
                    cv::Mat upInitFrame = frame.clone();
                    cv::resize(upInitFrame, upInitFrame, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
                    
                    cv::Mat kcfUpInitTmp;
                    if (upInitFrame.channels() == 1) cv::cvtColor(upInitFrame, kcfUpInitTmp, cv::COLOR_GRAY2BGR);
                    else kcfUpInitTmp = upInitFrame;

                    cv::Rect upscaledBbox(outBbox.x * 2, outBbox.y * 2, outBbox.width * 2, outBbox.height * 2);
                    m_tracker->init(kcfUpInitTmp, upscaledBbox);
                }
                else {
                    std::cout << "[Tracker] 현재 모드 유지 (크기 변화 없음)" << std::endl;
                    // 스케일 변경 조건이 아닐 때는 최신 스냅샷 저장 연산만 수행
                    m_targetTemplate = currentROI.clone(); // 신뢰도가 높을 때마다 템플릿 업데이트 (30프레임마다)
                    std::vector<cv::KeyPoint> kp;
                    m_targetDescriptors.release();
                    m_orb->detectAndCompute(currentROI, cv::noArray(), kp, m_targetDescriptors);
                }
            }
            
            // 실시간 신뢰도 평가 점수 계산
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

    // 소형 표적 대응을 위한 2배 업스케일링 유지
    cv::Mat verifiedRoi = currentROI.clone();
    if (currentROI.cols < 60) {
        cv::resize(currentROI, verifiedRoi, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
    }

    // [Step 1] NCC 점수 먼저 정직하게 측정
    float nccScore = 0.0f;
    if (!m_targetTemplate.empty()) {
        nccScore = calculateNCCConfidence(verifiedRoi);
    }

    // 🌟 [핵심 방어선 1]: 가림 현상 원천 차단
    // NCC 점수가 0.20 이하라는 것은 픽셀 겉모양이 완전히 바뀌었다(가려졌다)는 뜻입니다.
    // 이 경우 ORB가 아무리 만점을 뱉어도 무조건 "가짜 표적"으로 간주하고 즉시 튕겨냅니다.
    if (nccScore <= 0.20f) {
        std::cout << "[Verify Danger] NCC가 과락(0.00)입니다. 가림 현상 혹은 배경 오판으로 간주하여 즉시 기각합니다." << std::endl;
        return 0.0f; 
    }

    // [Step 2] ORB 점수 측정
    float orbScore = 0.0f;
    
    // 이 내부에 '매칭된 최종 특징점 개수'를 반환받거나 검사하는 로직이 있다면 베스트입니다.
    // 여기서는 안전하게 계산된 orbScore를 가져옵니다.
    orbScore = calculateORBConfidence(verifiedRoi); 

    std::cout << "[Verify Debug] Raw ORB: " << orbScore << " | Raw NCC: " << nccScore << std::endl;

    // [Step 3] 융합 점수 계산 (가중치 평균 구조)
    // max 구조를 버리고, 두 알고리즘이 교집합으로 동의할 때만 고득점을 주는 산술 구조 적용
    float finalScore = 0.0f;
    
    if (orbScore >= 0.40f) {
        // ORB와 NCC가 둘 다 양호할 때: NCC에 60%, ORB에 40%의 지분 부여 (텍스처 신뢰)
        finalScore = (orbScore * 0.4f) + (nccScore * 0.6f);
    } else {
        // ORB 특징점이 뭉개졌더라도 NCC 질감이 어느 정도 살아있다면 NCC 점수를 하향 조정하여 반영
        finalScore = nccScore * 0.7f;
    }

}