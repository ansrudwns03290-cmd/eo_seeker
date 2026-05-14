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

private:
    cv::Ptr<cv::Tracker> m_tracker; // 추적기 객체 포인터
    cv::Ptr<cv::ORB> m_orb;
    cv::Ptr<cv::DescriptorMatcher> m_matcher;

    cv::Mat m_targetDescriptors;

    cv::Rect m_lastBbox;
    float m_confidence;
    bool m_isInitialized;

    float verifyTarget(const cv::Mat& currentROI);
};