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
    struct TrackingResult {
        bool success;
        cv::Rect bbox;
        bool needTemplateUpdate;
        cv::Mat newTemplate;
        std::vector<cv::KeyPoint> newKeypoints;
        cv::Mat newDescriptors;

        // [Metrics] track_ms 세부 구간 계측 (2026-09 FPS 분석용).
        // update() 맨 끝에서만 채워지므로, 중간에 reinitTracker()가 result를
        // 통째로 덮어써도(result = reinitTracker(...)) 값이 사라지지 않는다.
        bool   is_upscaled = false; // 이번 프레임의 KCF 배율이 1보다 컸는지 (kcf_scale > 1)
        double resize_ms   = 0.0;   // 배율 != 1일 때 "풀프레임" resize+cvtColor 비용 (축소 포함, 배율 1이면 cvtColor만)
        double kcf_core_ms = 0.0;   // KCF 코어 m_tracker->update() 자체 비용
        double verify_ms   = 0.0;   // 5프레임마다 도는 verifyTarget(NCC+ORB) 비용 (해당 없으면 0)
        double periodic_ms = 0.0;   // 30프레임마다 도는 이진화/컨투어/ORB 재추출(+reinit) 비용 (해당 없으면 0)

        // [Metrics] EO-33/EO-5: KCF 입력 배율과 FFT 패치 크기 (resize=false 전제, 패치 = 2 × KCF bbox)
        float  kcf_scale   = 1.0f;
        int    kcf_patch_w = 0;
        int    kcf_patch_h = 0;
    };

    Tracker();
    ~Tracker();

    /**
     * @brief 추적기 초기화
     * @param frame 초기 프레임
     * @param bbox 획득 모듈에서 전달받은 초기 표적 영역
     * @return 초기화 성공 여부
     */
    bool init(const cv::Mat& frame, const cv::Rect& bbox, const cv::Mat& templateImg, const cv::Mat descriptors);

    /**
     * @brief 현재 프레임에서 표적 위치 업데이트
     * @param frame 현재 프레임
     * @param outBbox 업데이트된 표적 좌표 (출력 변수)
     * @return 추적 성공 시 true, 실패 시 false
     */
    TrackingResult update(const cv::Mat& frame, const cv::Mat& refTemplate, const cv::Mat& refDescriptors);

    float getConfidence() const { return m_confidence; }

    void setVerificationMode(bool useFeature) { m_useFeaturMode = useFeature; }

    /**
     * @brief 현재 시점의 outBbox를 기준으로 배율을 다시 계산하여 트래커 및 정답지 재설정
     * @param frame 원본 1배 해상도 프레임
     * @param outBbox 원본 해상도 기준 현재 표적 좌표
     * @param smallMode 소형 표적 모드 여부 (배율은 kcfgeom::computeScale로 재계산)
     */
    TrackingResult reinitTracker(const cv::Mat& frame, const cv::Rect& outBbox, bool smallMode,
                                  const cv::Rect& tightObjectRectRel = cv::Rect());

    float verifyCandidate(const cv::Mat& currentROI, const cv::Mat& refDescriptors, const cv::Mat& refTemplate);

private:
    cv::Ptr<cv::Tracker> m_tracker; // 추적기 객체 포인터
    cv::Ptr<cv::ORB> m_orb;
    cv::Ptr<cv::DescriptorMatcher> m_matcher;

    cv::Rect m_lastBbox;

    // [EO-33/EO-5] KCF 입력 좌표계 관리. 배율은 init/reinit 시점에만 계산한다.
    float    m_kcfScale  = 1.0f;   // KCF 입력 좌표 / 원본 좌표
    bool     m_smallMode = false;  // 소형 표적 정책(75/50 히스테리시스) 상태
    cv::Size m_origSize;           // 출력 bbox 크기 (init 시 원 ROI 크기로 고정)
    cv::Size m_kcfPatch;           // FFT 패치 크기 (= 2 × KCF bbox, init/reinit 시 고정)

    // 원본 프레임을 m_kcfScale만큼 리사이즈하고 KCF용 3채널로 맞춘다.
    cv::Mat makeKcfFrame(const cv::Mat& frame) const;
    // KCF 엔진을 resize=false로 만들고 원본 bbox로 시동한다 (init/reinit 공통).
    void startKcf(const cv::Mat& frame, const cv::Rect& origBox, bool smallMode);
    float m_confidence;
    bool m_isInitialized;
    bool m_useFeaturMode = false; // 특징점 매칭 모드 사용 여부

    int64_t m_frameCount = 0; // 프레임 카운터 (템플릿 업데이트 주기 관리용)

    // 표적 크기 기반 모드 스위칭(reinitTracker) 오탐 방지용:
    // 노이즈성 컨투어 측정 한 번만으로 KCF를 통째로 재시작하지 않도록,
    // 같은 판정이 연속으로 나올 때만 실제 전환을 수행한다.
    int m_smallSizeStreak = 0;
    int m_largeSizeStreak = 0;
    static const int REINIT_CONFIRM_STREAK = 2;

    float verifyTarget(const cv::Mat& currentROI, const cv::Mat& refTemplate, const cv::Mat& refDescriptors);
    float calculateORBConfidence(const cv::Mat& currentROI, const cv::Mat& refDescriptors);
    float calculateNCCConfidence(const cv::Mat& currentROI, const cv::Mat& refTemplate);

    // reinitTracker 직전 안전장치: outBbox가 직전 프레임(m_lastBbox) 대비
    // 한 프레임 만에 비정상적으로 멀리 튀었다면(KCF 오검출 가능성) 재시작을 보류한다.
    bool isReinitPositionSane(const cv::Rect& newBox) const;
};
