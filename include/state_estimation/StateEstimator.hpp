#pragma once

#include <opencv2/opencv.hpp>

class StateEstimator {
public:
    StateEstimator();
    ~StateEstimator() = default;

    /**
     * @brief 칼만 필터를 특정 위치와 속도로 초기화합니다. (최초 획득 또는 REACQUIRE 시 사용)
     * @param initial_pos 최초 표적의 중심 좌표 (x, y)
     */
    void initialize(const cv::Point2f& initial_pos);

    /**
     * @brief 물리 법칙(이전 속도)에 기반하여 현재 프레임의 상태를 예측합니다.
     * @param dt 이전 프레임과 현재 프레임 사이의 시간 간격 (seconds)
     * @return cv::Point2f 예측된 표적의 중심 좌표
     */
    cv::Point2f predict(double dt);

    /**
     * @brief KCF 추적기의 실제 측정값을 반영하여 상태를 보정(Update)합니다.
     * @param measurement_pos KCF 추적기가 검출한 표적의 중심 좌표
     * @return cv::Point2f 보정 완료된 최적의 표적 중심 좌표
     */
    cv::Point2f update(const cv::Point2f& measurement_pos);

    // --- Getters ---
    cv::Point2f getEstimatedPosition() const;
    cv::Point2f getEstimatedVelocity() const;
    bool isInitialized() const { return m_is_initialized; }
    
    /**
     * @brief 추적 유실(LOST) 상태일 때, 외부(FSM 등)에서 호출하여 초기화 플래그를 해제합니다.
     */
    void reset();

private:
    cv::KalmanFilter m_kf;
    bool m_is_initialized;

    // 공분산 행렬 설정을 위한 튜닝 파라미터 계수
    const float m_process_noise_coef;  // 시스템(물리 모델) 불확실성 가중치
    const float m_measure_noise_coef;  // 센서(KCF 측정) 노이즈 가중치
};