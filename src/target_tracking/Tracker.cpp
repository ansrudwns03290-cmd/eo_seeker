#include "target_tracking/Tracker.hpp"
#include <iostream>
#include <algorithm>

Tracker::Tracker() 
    : m_confidence(0.0f), m_isInitialized(false) {
    // 생성자에서는 객체를 할당하지 않고 init 호출 시 할당하는 것이 메모리 관리에 유리합니다.
    m_orb = cv::ORB::create(150);
    // ORB는 Binary 기술자이므로 Hamming 거리를 사용하는 매칭기를 생성합니다.
    m_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

Tracker::~Tracker() {}

/**
 * @brief KCF 추적기 초기화
 * @param frame 초기 프레임 이미지
 * @param bbox 초기 표적 영역 (AcquisitionManager에서 전달받은 ROI)
 * @return 초기화 성공 여부
 */
bool Tracker::init(const cv::Mat& frame, const cv::Rect& bbox, const cv::Mat& templateImg, const cv::Mat descriptors) {
    if (frame.empty() || bbox.width <= 0 || bbox.height <= 0) {
        std::cerr << "[Tracker] Invalid Init Data!" << std::endl;
        return false;
    }

    try {
        cv::Rect safeRoi = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safeRoi.width <= 0 || safeRoi.height <= 0) return false;
        
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
Tracker::TrackingResult Tracker::update(const cv::Mat& frame, const cv::Mat& refTemplate, const cv::Mat& refDescriptors) {
    // 1. 결과 데이터 구조체 초기화
    TrackingResult result;
    result.needTemplateUpdate = false;
    result.success = false;
    result.bbox = cv::Rect(0, 0, 0, 0);

    // 2. 초기화 상태 및 프레임 유효성 검사
    if (!m_isInitialized || frame.empty()) {
        return result;
    }
    
    cv::Mat workingFrame = frame.clone();
    cv::Rect virtualBbox;

    // 3. 적응형 해상도 스케일링 (Small Target 처리)
    // 표적이 작을 경우(60px 미만) 2배 확대하여 추적 성능을 높이는 모드 적용
    bool isUpscaledMode = (m_lastBbox.width < 60);

    if (isUpscaledMode) {
        cv::resize(workingFrame, workingFrame, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
        virtualBbox = m_lastBbox;
        virtualBbox.x *= 2;
        virtualBbox.y *= 2;
        virtualBbox.width *= 2;
        virtualBbox.height *= 2;
    } else {
        virtualBbox = m_lastBbox;
    }
    
    // 4. 추적기 입력 포맷 정합 (KCF의 경우 3채널 입력 필요)
    cv::Mat kcfInputFrame;
    if (workingFrame.channels() == 1) {
        cv::cvtColor(workingFrame, kcfInputFrame, cv::COLOR_GRAY2BGR);
    } else {
        kcfInputFrame = workingFrame;
    }

    // 5. KCF 추적 엔진 수행 (핵심 추적)
    bool success = m_tracker->update(kcfInputFrame, virtualBbox);
    result.success = success;

    if (success) {
        // 6. 좌표계 복원 (스케일링 모드였다면 원본 해상도로 환산)
        cv::Rect outBbox;
        m_frameCount++; 
        
        if (isUpscaledMode) {
            outBbox.x = virtualBbox.x / 2;
            outBbox.y = virtualBbox.y / 2;
            outBbox.width = virtualBbox.width / 2;
            outBbox.height = virtualBbox.height / 2;
        } else {
            outBbox = virtualBbox;
        }

        result.bbox = outBbox;
        
        // 7. 추적 신뢰도 평가 및 모델 갱신 로직 (주기적 수행)
        // 수정
        cv::Rect safeRoi = virtualBbox & cv::Rect(0, 0, kcfInputFrame.cols, kcfInputFrame.rows);
        
        if (safeRoi.width > 0 && safeRoi.height > 0) {
            cv::Mat currentROI = kcfInputFrame(safeRoi).clone();
            
            // a. 현재 추적 위치의 신뢰도 계산
            if (m_frameCount % 5 == 0) {
                m_confidence = verifyTarget(currentROI, refTemplate, refDescriptors);
            }
            
            // b. 30프레임 주기마다 수행하는 정밀 모델 검증 및 업데이트
            if (m_frameCount % 30 == 0) { 
                cv::Mat grayROI;
                
                // 1) 전처리: 템플릿 갱신을 위한 흑백 변환
                if (currentROI.channels() == 3) {
                    cv::cvtColor(currentROI, grayROI, cv::COLOR_BGR2GRAY);
                } else {
                    grayROI = currentROI.clone(); 
                }

                // 2) 이진화 및 노이즈 제거
                cv::Mat binImg;
                cv::threshold(grayROI, binImg, 70, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); 
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                cv::morphologyEx(binImg, binImg, cv::MORPH_OPEN, kernel);

                // 3) 외곽선 기반 표적 실측
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(binImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                int maxObjectWidth = 0;
                cv::Rect maxContourBox = cv::Rect(0, 0, 0, 0);

                for (const auto& contour : contours) {
                    cv::Rect objectBox = cv::boundingRect(contour);
                    if (objectBox.width > maxObjectWidth && objectBox.width > 10) {
                        maxObjectWidth = objectBox.width;
                        maxContourBox = objectBox;
                    }
                }

                // 4) 표적 크기 기반 모드 스위칭 (Adaptive Mode Management)
                int currentOriginalWidth = isUpscaledMode ? (maxObjectWidth / 2) : maxObjectWidth; 
                if (currentOriginalWidth <= 0) currentOriginalWidth = outBbox.width;
        
                if (m_confidence > 0.6f) {
                    // 표적 커지면 원본 모드로, 작아지면 업스케일 모드로 전환
                    if (isUpscaledMode && currentOriginalWidth >= 75) {
                        std::cout << "[Tracker] 표적 크기 75px 도달! 원본 모드로 전환" << std::endl;
                        result = reinitTracker(frame, outBbox, 1.0f);
                        result.bbox = outBbox;
                        result.success = true;
                    }
                    else if (!isUpscaledMode && currentOriginalWidth < 50) {
                        std::cout << "[Tracker] 표적 크기 50px 미만! 업스케일 모드로 전환" << std::endl;
                        result = reinitTracker(frame, outBbox, 2.0f);
                        result.bbox = outBbox;
                        result.success = true;
                    }
                    else {
                        // 5) 안정적인 추적 중이면 템플릿/특징점 갱신 Trigger
                        std::cout << "[Tracker] 현재 모드 유지 (크기 변화 없음)" << std::endl;
                    
                        cv::Mat newTemplate;
                        if (maxContourBox.width > 10 && maxContourBox.height > 10) {
                            cv::Rect tightRoi = maxContourBox & cv::Rect(0, 0, currentROI.cols, currentROI.rows);
                            newTemplate = currentROI(tightRoi).clone();
                        } else {
                            newTemplate = currentROI.clone();
                        }

                        // ORB 특징점 재추출
                        std::vector<cv::KeyPoint> kp;
                        cv::Mat newDescriptors;
                        m_orb->detectAndCompute(currentROI, cv::noArray(), kp, newDescriptors);
                        
                        // 결과 구조체에 업데이트 정보 할당
                        result.needTemplateUpdate = true;
                        result.newTemplate = newTemplate;
                        result.newKeypoints = kp;
                        result.newDescriptors = newDescriptors;

                        std::cout << "[Tracker] Template Update Triggered. New ORB Keypoints: " << kp.size() << std::endl;
                    }
                }
            }
        } else {
            m_confidence = 0.0f; // ROI 확보 실패 시 신뢰도 0
        }

        m_lastBbox = outBbox; // 최종 추적 성공 위치 업데이트
    } else {
        // 8. 추적 실패 처리
        m_confidence = 0.0f;
        m_isInitialized = false; // 추적기 초기화 상태 해제 (상위 FSM에 재획득 유도)
        std::cout << "[Tracker] Target LOST" << std::endl;
    }

    return result;
}

/**
 * @brief 현재 추적된 표적의 신뢰도를 검증
 * @param currentROI 현재 추적된 표적 영역
 * @return 신뢰도 점수
 */
float Tracker::verifyTarget(const cv::Mat& currentROI, const cv::Mat& refTemplate, const cv::Mat& refDescriptors) {
    if (currentROI.empty()) return 0.0f;

    // 1. 특징점 데이터가 있다면 ORB 시도
    const int MIN_DESCRIPTOR_COUNT = 7; 
    
    if (refDescriptors.rows >= MIN_DESCRIPTOR_COUNT) {
        float orbScore = calculateORBConfidence(currentROI, refDescriptors);
        
        // ORB 매칭 결과가 유의미하다면 즉시 반환
        if (orbScore > 0.1f) return orbScore; 
    }

    // 2. 특징점이 없거나 ORB 점수가 너무 낮으면 NCC 시도
    if (!refTemplate.empty()) {
        return calculateNCCConfidence(currentROI, refTemplate);
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
float Tracker::calculateORBConfidence(const cv::Mat& currentROI, const cv::Mat& refDescriptors) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    m_orb->detectAndCompute(currentROI, cv::noArray(), keypoints, descriptors);

    if (descriptors.empty()) return 0.0f;

    std::vector<cv::DMatch> matches;
    m_matcher->match(refDescriptors, descriptors, matches);

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
float Tracker::calculateNCCConfidence(const cv::Mat& currentROI, const cv::Mat& refTemplate) {
    if (refTemplate.empty() || currentROI.empty()) return 0.0f;

    cv::Mat res, resizedROI;
    cv::resize(currentROI, resizedROI, refTemplate.size());
    
    // NCC 매칭을 위해 입력 이미지와 템플릿의 채널 수가 다르면, 안전하게 맞춰주는 전처리 단계
    if (resizedROI.channels() != refTemplate.channels()) {
        if (refTemplate.channels() == 1 && resizedROI.channels() == 3) {
            // 정답지가 흑백인데 입력이 컬러라면, 입력을 흑백으로 변환
            cv::cvtColor(resizedROI, resizedROI, cv::COLOR_BGR2GRAY);
        } 
        else if (refTemplate.channels() == 3 && resizedROI.channels() == 1) {
            // 정답지가 컬러인데 입력이 흑백이라면, 입력을 가짜 컬러로 확장
            cv::cvtColor(resizedROI, resizedROI, cv::COLOR_GRAY2BGR);
        }
    }

    static int cnt = 0;
    cv::matchTemplate(resizedROI, refTemplate, res, cv::TM_CCOEFF_NORMED);
    // cv::imwrite("C:/eo_seeker/debug_images/resized_roi" + std::to_string(cnt++) + ".png", resizedROI); // 디버그용 후보 이미지 저장
    // cv::imwrite("C:/eo_seeker/debug_images/target_template" + std::to_string(cnt) + ".png", refTemplate); // 디버그용 후보 이미지 저장
        
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

Tracker::TrackingResult Tracker::reinitTracker(const cv::Mat& frame, const cv::Rect& outBbox, float scaleFactor) {
    TrackingResult result;
    result.needTemplateUpdate = true;
    
    // 1. 공통 안전 영역 도려내기 및 1배 원본 정답지 스냅샷 백업
    cv::Rect origSafeRoi = outBbox & cv::Rect(0, 0, frame.cols, frame.rows);
    result.newTemplate = frame(origSafeRoi).clone();

    // 2. 공통 ORB 기술자 데이터 업데이트 (1배 원본 조각 기준 생성)
    std::vector<cv::KeyPoint> kp;
    m_orb->detectAndCompute(result.newTemplate, cv::noArray(), kp, result.newDescriptors);
    std::cout << "[Tracker:Reset] New ORB descriptors : " << result.newDescriptors.rows << " points" << std::endl;

    // 3. 기존 KCF 추적기 구형 기억 파괴 및 재생성
    m_tracker.release();
    m_tracker = cv::TrackerKCF::create();

    // 4. 스케일 팩터(scaleFactor)에 따른 가공 및 KCF 시동 분기
    cv::Mat kcfInitFrame;
    cv::Rect kcfInitBbox;

    if (std::abs(scaleFactor - 2.0f) < 0.01f) {
        // [2배 업스케일링 모드 진입인 경우]
        cv::Mat upScaledFrame = frame.clone();
        cv::resize(upScaledFrame, upScaledFrame, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
        kcfInitFrame = upScaledFrame;

        kcfInitBbox = cv::Rect(outBbox.x * 2, outBbox.y * 2, outBbox.width * 2, outBbox.height * 2);
    } else {
        // [1배 원본 모드 진입 혹은 유지인 경우]
        kcfInitFrame = frame.clone();
        kcfInitBbox = origSafeRoi;
    }

    // 5. KCF 고정 채널 안전장치 (가짜 3채널 복제 래핑)
    if (kcfInitFrame.channels() == 1) {
        cv::cvtColor(kcfInitFrame, kcfInitFrame, cv::COLOR_GRAY2BGR);
    }

    // 6. 완벽하게 해상도가 정합된 공간에서 KCF 새출발
    m_tracker->init(kcfInitFrame, kcfInitBbox);
    m_lastBbox = outBbox; // 1배 원본 좌표 박제로 동역학 싱크 수립

    return result;
}

float Tracker::verifyCandidate(const cv::Mat& currentROI, const cv::Mat& refDescriptors, const cv::Mat& refTemplate) {
    if (currentROI.empty()) return 0.0f;

    // 소형 표적 대응을 위한 2배 업스케일링 유지
    cv::Mat verifiedRoi = currentROI.clone();
    if (currentROI.cols < 60) {
        cv::resize(currentROI, verifiedRoi, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
    }

    // [Step 1] NCC 점수 먼저 정직하게 측정
    float nccScore = 0.0f;
    if (!refTemplate.empty()) {
        nccScore = calculateNCCConfidence(verifiedRoi, refTemplate);
    }

    // [Step 2] ORB 점수 측정
    float orbScore = calculateORBConfidence(verifiedRoi, refDescriptors);

    float finalScore = 0.0f;

    int baseDescriptorCount = refDescriptors.rows;

    if (baseDescriptorCount < 7) {
        // 특징점 수 너무 적으면 NCC만 최종 점수에 반영
        finalScore = nccScore;

        std::cout << "[Tracker: Verify] ORB 배제 (특징점 " << baseDescriptorCount
                    << "개로 부족) - NCC 점수만 반영. Score: " << finalScore << std::endl;
    } else{
        //특징점 풍부하다면 융합 논리 적용
        if (nccScore <= 0.20f) {
            std::cout << "[Tracker: Verify] NCC 과락. 가림 현상 기각." << std::endl;
            return 0.0f;
        }

        if (orbScore >= 0.40f) {
            finalScore = (orbScore * 0.4f) + (nccScore * 0.6f);
        } else {
            // ORB 점수 낮아도 NCC 점수 70% 인정
            finalScore = nccScore * 0.7f;
        }

        std::cout << "[Tracker: Verify] ORB Score: " << orbScore
                    << ", NCC Score: " << nccScore
                    << " -> Final Confidence: " << finalScore << std::endl;
    }

    return finalScore;
}