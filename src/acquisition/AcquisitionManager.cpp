#include "acquisition/AcquisitionManager.hpp"
#include "target_tracking/Tracker.hpp"
#include "common/DebugConfig.hpp"

AcquisitionManager::AcquisitionManager() {
    // ORB 특징점 추출기 초기화
    m_orb = cv::ORB::create(150, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 10); 
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

    // [디버그] 물체 분리에 쓰인 원본 크롭과 Otsu 이진화 마스크 저장.
    // ORB Keypoints가 0으로 나오는 문제를 진단할 때, 여기서 물체 영역이 제대로
    // 분리됐는지(마스크가 텅 비었거나 표적과 무관한 영역을 잡았는지) 눈으로 바로 확인 가능.
    if (DebugConfig::kEnableImageDump) {
        cv::imwrite(DebugConfig::kDebugImageDir + "init_roi_raw.png", roiImg);
        cv::imwrite(DebugConfig::kDebugImageDir + "init_binary_mask.png", binary);
    }

    // 2. 물체 윤곽선 추출 및 정밀 모델링
    cv::Mat featureSourceImg; // [디버그용] ORB 추출에 실제로 들어간 이미지를 branch와 무관하게 기록
    cv::Rect actualObjectRect;
    if (findLargestObject(binary, actualObjectRect)) {
        cv::Mat objectOnly = roiImg(actualObjectRect);
        m_targetTemplate = buildTemplate(roiImg);

        cv::Mat objectGray;
        if (objectOnly.channels() == 3)
            cv::cvtColor(objectOnly, objectGray, cv::COLOR_BGR2GRAY);
        else
            objectGray = objectOnly;
        m_orb->detectAndCompute(objectGray, cv::noArray(), m_targetKeypoints, m_targetDescriptors);
        m_targetRatio = static_cast<double>(actualObjectRect.width) / actualObjectRect.height;

        m_isFeatureRich = (m_targetKeypoints.size() >= 10);

        std::cout << "[Acquisition] Mode: " << (m_isFeatureRich ? "ORB-Rich" : "Template-Only") << std::endl;
        featureSourceImg = objectGray;
    } else {
        // 물체 분리 실패 시 박스 전체 사용
        m_targetTemplate = buildTemplate(roiImg); // gray 변환만 수행
        m_orb->detectAndCompute(roiImg, cv::noArray(), m_targetKeypoints, m_targetDescriptors);
        m_targetRatio = static_cast<double>(roiImg.cols) / roiImg.rows;
        std::cout << "[Acquisition] Target Registered (Box Mode)" << std::endl;
        featureSourceImg = roiImg;
    }

    // 드리프트 검증용 원본 스냅샷 보관 (updateTargetModel에서 절대 덮어쓰지 않음)
    m_originalTemplate = m_targetTemplate.clone();
    m_originalDescriptors = m_targetDescriptors.clone();

    std::cout << "[Acquisition: Debug] Extracted ORB Keypoints: " << m_targetKeypoints.size() << std::endl;

    // [디버그] 최종 NCC 템플릿 + ORB 추출에 실제로 쓰인 이미지 + 키포인트 시각화 저장
    if (DebugConfig::kEnableImageDump) {
        cv::imwrite(DebugConfig::kDebugImageDir + "init_template.png", m_targetTemplate);
        if (!featureSourceImg.empty()) {
            cv::imwrite(DebugConfig::kDebugImageDir + "init_orb_source.png", featureSourceImg);

            cv::Mat kpVis;
            cv::drawKeypoints(featureSourceImg, m_targetKeypoints, kpVis, cv::Scalar(0, 255, 0));
            cv::imwrite(DebugConfig::kDebugImageDir + "init_orb_keypoints.png", kpVis);
        }
    }
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
 * @brief 표적 크기에 따라 패딩을 적용한 후 gray 변환된 템플릿 반환
 */
cv::Mat AcquisitionManager::buildTemplate(const cv::Mat& sourceImg, const cv::Rect& objectRect) {
    // 표적 크기 기반 패딩 결정
    int padding = (objectRect.width < 40 || objectRect.height < 40) ? 10 : 5;

    cv::Rect paddedRect;
    paddedRect.x      = std::max(0, objectRect.x - padding);
    paddedRect.y      = std::max(0, objectRect.y - padding);
    paddedRect.width  = std::min(sourceImg.cols - paddedRect.x, objectRect.width  + padding * 2);
    paddedRect.height = std::min(sourceImg.rows - paddedRect.y, objectRect.height + padding * 2);

    cv::Mat cropped = sourceImg(paddedRect).clone();

    cv::Mat grayTemplate;
    if (cropped.channels() == 3)
        cv::cvtColor(cropped, grayTemplate, cv::COLOR_BGR2GRAY);
    else
        grayTemplate = cropped;

    return grayTemplate;
}

/**
 * @brief 이미 crop된 이미지에서 gray 변환만 수행 (updateTargetModel용 오버로드)
 */
cv::Mat AcquisitionManager::buildTemplate(const cv::Mat& croppedImg) {
    cv::Mat grayTemplate;
    if (croppedImg.channels() == 3)
        cv::cvtColor(croppedImg, grayTemplate, cv::COLOR_BGR2GRAY);
    else
        grayTemplate = croppedImg.clone();
    return grayTemplate;
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

    if (searchRoi.width <= 0 || searchRoi.height <= 0) {
        std::cout << "[Acquisition LOST] 탐색 영역이 화면 밖. searchRoi: "
                    << searchRoi << std::endl;
        return false;
    }

    cv::Mat croppedSearchImg = processedGrayImg(searchRoi);

    // [디버그] LOST 상태에서 실제로 탐색한 영역과, 그 시점에 쓰이고 있던 NCC 템플릿을 저장.
    // 매 프레임 호출되므로 프레임 카운터를 붙여 시간에 따른 변화를 순서대로 볼 수 있게 한다.
    static int s_lostSearchCallCount = 0;
    s_lostSearchCallCount++;
    if (DebugConfig::kEnableImageDump) {
        cv::imwrite(DebugConfig::kDebugImageDir + "lost_search_window_" + std::to_string(s_lostSearchCallCount) + ".png",
                    croppedSearchImg);
        if (!m_targetTemplate.empty()) {
            cv::imwrite(DebugConfig::kDebugImageDir + "lost_current_template.png", m_targetTemplate);
        }
    }

    // 템플릿 매칭 수행
    cv::Mat matchResult;
    cv::Mat currentTemplate = m_targetTemplate; // 템플릿이 없는 경우 작은 검은 이미지로 대체
    if(m_targetTemplate.empty()) {
        std::cout << "[DEBUG] Warning: Target template is empty. Using placeholder for matching." << std::endl;
        return false;
    }

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

        const int MIN_DESCRIPTOR_COUNT = 7;
        bool verified = (m_targetDescriptors.rows < MIN_DESCRIPTOR_COUNT) ? true : verifyCandidateWithORB(candidateROI);

        // [디버그] NCC 최고점을 넘긴 후보 위치를 탐색 창 위에 박스로 표시해서 저장.
        // ORB 검증까지 통과했는지 여부를 파일명에 남겨, 실제 표적을 잡았는데 ORB 검증에서
        // 걸러진 건지 아니면 애초에 엉뚱한 곳을 잡은 건지 구분할 수 있게 한다.
        if (DebugConfig::kEnableImageDump) {
            cv::Mat candidateVis;
            cv::cvtColor(croppedSearchImg, candidateVis, cv::COLOR_GRAY2BGR);
            cv::rectangle(candidateVis, bestCandidate, cv::Scalar(0, 255, 0), 2);
            cv::imwrite(DebugConfig::kDebugImageDir + "lost_best_candidate_" + std::to_string(s_lostSearchCallCount)
                            + (verified ? "_verified" : "_rejected") + ".png",
                        candidateVis);
        }

        if (verified) {
            // 전체 좌표계로 변환
            outCandidateBox.x = searchRoi.x + bestCandidate.x;
            outCandidateBox.y = searchRoi.y + bestCandidate.y;
            outCandidateBox.width = bestCandidate.width;
            outCandidateBox.height = bestCandidate.height;
            return true;
        }
    }

    std::cout << "[Acquisition LOST] NCC 최고 점수: " << maxVal
                << ", Threshold: " << MATCH_THRESHOLD  << std::endl;
    
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

    if (good_matches < 5) {
        std::cout << "[Acquisition LOST] ORB 검증 실패. Good matches: " << good_matches
                    << ", Threshold: 5" << std::endl;
    }
    return (good_matches >= 5);
}

/**
 * @brief 주기 갱신으로 들어온 새 템플릿/디스크립터가 최초 등록된 원본 표적과
 *        여전히 충분히 비슷한지 검증한다 (템플릿 드리프트 방지용 anchor 체크).
 *        직전 기준과만 비교하면, KCF가 배경으로 조금씩 밀려도 매번 "직전과 비슷하다"는
 *        이유로 계속 통과해버려 결국 완전히 다른 것을 표적으로 착각하게 된다.
 *        그래서 절대 바뀌지 않는 최초 원본과도 항상 같이 비교한다.
 */
bool AcquisitionManager::isStillSimilarToOriginal(const cv::Mat& candidateTemplate, const cv::Mat& candidateDescriptors) const {
    if (m_originalTemplate.empty() && m_originalDescriptors.empty()) {
        return true; // 비교 기준 자체가 없는 비정상 상황이면 통과시킨다
    }

    // 1. NCC 기반 비교 (원본 템플릿 대비)
    bool nccChecked = false;
    bool nccPass = false;
    if (!m_originalTemplate.empty() && !candidateTemplate.empty()) {
        cv::Mat resized;
        cv::resize(candidateTemplate, resized, m_originalTemplate.size());

        if (resized.channels() != m_originalTemplate.channels()) {
            if (m_originalTemplate.channels() == 1 && resized.channels() == 3) {
                cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY);
            } else if (m_originalTemplate.channels() == 3 && resized.channels() == 1) {
                cv::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
            }
        }

        cv::Mat res;
        cv::matchTemplate(resized, m_originalTemplate, res, cv::TM_CCOEFF_NORMED);
        double minVal, maxVal;
        cv::minMaxLoc(res, &minVal, &maxVal);

        nccChecked = true;
        // Tracker::calculateNCCConfidence의 완전 통과선(0.48)보다는 관대하고
        // 완전 실패선(0.25)보다는 엄격한 중간 지점을 anchor 기준으로 사용
        nccPass = (maxVal >= 0.35);

        std::cout << "[Acquisition: Anchor] 원본 대비 NCC: " << maxVal
                    << " (기준 0.35, " << (nccPass ? "통과" : "미달") << ")" << std::endl;
    }

    // 2. ORB 기반 비교 (원본 디스크립터 대비)
    bool orbChecked = false;
    bool orbPass = false;
    const int MIN_DESCRIPTOR_COUNT = 5;
    if (m_originalDescriptors.rows >= MIN_DESCRIPTOR_COUNT && !candidateDescriptors.empty()) {
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        matcher.match(m_originalDescriptors, candidateDescriptors, matches);

        int goodMatches = 0;
        for (const auto& m : matches) {
            if (m.distance < 80.0) goodMatches++;
        }

        orbChecked = true;
        orbPass = (goodMatches >= 3);

        std::cout << "[Acquisition: Anchor] 원본 대비 ORB Good Matches: " << goodMatches
                    << " (기준 3, " << (orbPass ? "통과" : "미달") << ")" << std::endl;
    }

    // 3. 판정: 검증 가능했던 항목 중 하나라도 통과하면 인정한다.
    //    (배경 변화 없이도 조명/각도 변화만으로 한쪽 지표가 흔들릴 수 있어, 너무 엄격하게
    //     둘 다 요구하면 정상적인 갱신까지 막아버릴 수 있다.)
    //    단, 검증 가능한 항목이 하나도 없었다면 보수적으로 통과시킨다.
    if (!nccChecked && !orbChecked) return true;
    return (nccChecked && nccPass) || (orbChecked && orbPass);
}

void AcquisitionManager::updateTargetModel(const cv::Mat& newTemplate, const cv::Mat& newDescriptors) {
    if (newTemplate.empty() || newDescriptors.empty()) {
        std::cerr << "[Acquisition] Warning: Attempting to update target model with empty template or descriptors." << std::endl;
        return;
    }

    if (!isStillSimilarToOriginal(newTemplate, newDescriptors)) {
        std::cout << "[Acquisition] 갱신 거부: 최초 등록된 원본 표적과 너무 달라짐 (드리프트/배경 오검출 의심)" << std::endl;
        return;
    }

    m_targetTemplate = buildTemplate(newTemplate); // gray 변환 (crop 없이 변환만 수행)
    m_targetDescriptors = newDescriptors.clone();
    std::cout << "[Acquisition] Target model updated with new template and descriptors." << std::endl;
}