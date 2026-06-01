#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>
#include "common/Frame.hpp"

/**
 * @brief KCF 알고리즘을 이용한 객체 추적 클래스
 */
class Tracker {
public:
    Tracker();
    ~Tracker();

    /**
     * @brief 추적기 초기화
     * @param frame 초기 프레임
     * @param bbox 획득 모듈에서 전달받은 초기 표적 영역
     * @return 초기화 성공 여부
     */
    bool init(const cv::Mat& frame, const cv::Rect& bbox);

    /**
     * @brief AcquisitionManager에서 추출한 표적의 ORB 기술자를 전달받음
     */
    void setTargetDescriptors(const cv::Mat& descriptors);

    /**
     * @brief 현재 프레임에서 표적 위치 업데이트
     * @param frame 현재 프레임
     * @param outBbox 업데이트된 표적 좌표 (출력 변수)
     * @return 추적 성공 시 true, 실패 시 false
     */
    bool update(const cv::Mat& frame, cv::Rect& outBbox);

    float getConfidence() const { return m_confidence; }

    void setVerificationMode(bool useFeature) { m_useFeaturMode = useFeature; }
    void setTargetTemplate(const cv::Mat& templateImg) { m_targetTemplate = templateImg.clone(); }

    /**
     * @brief REACQUIRE 상태에서 후보 검증
     * @param currentROI 현재 후보 영역의 이미지 조각
     * @return 후보의 신뢰도 점수 (0.0 ~ 1.0)
     */
    float verifyCandidate(const cv::Mat& currentROI);

    /**
     * @brief 현재 시점의 outBbox를 기준으로 스케일을 반영하여 트래커 및 정답지 재설정
     * @param frame 원본 1배 해상도 프레임
     * @param outBbox 원본 해상도 기준 현재 표적 좌표
     * @param scaleFactor 업스케일링 여부에 따른 스케일 팩터 (예: 2.0f)
     */
    void reinitTracker(const cv::Mat& frame, const cv::Rect& outBbox, float scaleFactor);

private:
    cv::Ptr<cv::Tracker> m_tracker; // 추적기 객체 포인터
    cv::Ptr<cv::ORB> m_orb;
    cv::Ptr<cv::DescriptorMatcher> m_matcher;

    cv::Mat m_targetDescriptors;

    cv::Rect m_lastBbox;
    float m_confidence;
    bool m_isInitialized;
    bool m_useFeaturMode = false; // 특징점 매칭 모드 사용 여부
    cv::Mat m_targetTemplate; // 특징점 없을 때 사용할 표적 템플릿 이미지

    int64_t m_frameCount = 0; // 프레임 카운터 (템플릿 업데이트 주기 관리용)

    float verifyTarget(const cv::Mat& currentROI);
    float calculateORBConfidence(const cv::Mat& currentROI);
    float calculateNCCConfidence(const cv::Mat& currentROI);
};