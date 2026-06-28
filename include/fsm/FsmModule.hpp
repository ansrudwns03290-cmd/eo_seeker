#pragma once

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <string>
#include <deque>
#include "common/Frame.hpp"

// 시스템의 4가지 핵심 상태 정의
enum class FSMState {
    SEARCH,       // 전체 화면 탐색 및 대기
    TRACK,        // 정상 연속 추적 및 서보 제어
    LOST,         // 상실 판정 및 정적 데이터 봉인 (경량 스캔)
    REACQUIRE     // 후보 정밀 검증 및 추적기 재부팅
};

class FsmModule {
public:
    FsmModule();
    ~FsmModule();

    /**
     * @brief 매 프레임 FSM 상태를 갱신하는 제어 루프
     * @param currentFrame 현재 프레임 데이터 (영상, 타임스탬프 등)
     * @param trackerConfidence KCF 추적기의 현재 신뢰도
     * @param kcfSuccess KCF 추적기의 현재 프레임에서의 성공 여부
     */
    void update(const Frame& currentFrame, float trackerConfidence, bool kcfSuccess, const cv::Point2f& currentVelocity);

    /**
     * @brief 현재 FSM 상태 반환(제어, 시각화 등에서 참조용)
     */
    FSMState getCurrentState() const;
    std::string getStateString() const;

    /**
     * @brief 강제 초기화를 위한 setter(예외 처리 및 리셋용)
     */
    void forceSetState(FSMState newState);

    /**
     * @brief FSM 내부에서 관리하는 타겟 박스 좌표 반환 (출력용)
      * @return cv::Rect 현재 FSM이 보증하는 최종 표적 좌표
     */
    cv::Rect getTargetBox() const;
    void setTargetBox(const cv::Rect& box);

    int getSearchWindowSize() const { return m_isHighSpeedLoss ? 160 : 80; } // 고속 손실 시 더 넓은 탐색 윈도우 사용

private:
    FSMState m_currentState;    // 현재 시스템 상태
    cv::Rect m_targetBox;       // FSM이 보증하는 최종 표적 좌표
    cv::Rect m_temporaryBox;    // REACQUIRE 상태에서 정밀 검증할 임시 후보 박스
    std::deque<float> m_confHistory; // 최근 N프레임 신뢰도 기록 (LOST 상태 판단용)
    
    // 타이머 및 누적 카운터 변수
    int64_t m_lostStartTime;    // LOST 상태에 진입한 최초 시점의 타임스탬프 (ms)
    int m_lowConfidenceCounter; // TRACK 상태에서 신뢰도 저하 누적 프레임 카운터

    // 고속 기동 예외 처리 플래그
    bool m_isHighSpeedLoss;     // 놓치기 직전 속도가 빨랐는지 여부
    
    // 내부 상태 전이 로깅 함수
    void logTransition(FSMState fromState, FSMState toState);
};