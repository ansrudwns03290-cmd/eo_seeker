#pragma once

#include <string>
#include "common/ServoCommand.hpp"

/**
 * @brief 표적 중심 오차를 기반으로 Pan/Tilt 제어 명령을 생성하는 클래스
 */
class ControlCommand {
public:
    /**
     * @brief 제어 명령 생성 모듈 생성자
     * @param p_gain_x Pan 방향 P 게인
     * @param p_gain_y Tilt 방향 P 게인
     * @param d_gain_x Pan 방향 D 게인 (기본값 0)
     * @param d_gain_y Tilt 방향 D 게인 (기본값 0)
     */
    ControlCommand(double p_gain_x, double p_gain_y, double d_gain_x = 0.0, double d_gain_y = 0.0);
    
    ~ControlCommand() = default;

    /**
     * @brief 오차를 입력받아 서보 제어 명령을 계산하는 핵심 함수
     * @param target_x 표적의 X 좌표 (Kalman 예측치 또는 Tracker 중심)
     * @param target_y 표적의 Y 좌표 (Kalman 예측치 또는 Tracker 중심)
     * @param fsm_state 현재 시스템의 FSM 상태명 ("TRACK", "LOST" 등)
     * @return ServoCommand 계산된 제어 명령 구조체
     */
    ServoCommand calculateCommand(double target_x, double target_y, const std::string& fsm_state);

    // --- 게인 및 파라미터 제어 인터페이스 ---
    void setPGains(double p_x, double p_y);
    void setDGains(double d_x, double d_y);
    void setDeadZone(double pixels);
    void setOutputLimits(double pan_min, double pan_max, double tilt_min, double tilt_max);

private:
    // 제어 게인 파라미터
    double kp_x_;
    double kp_y_;
    double kd_x_;
    double kd_y_;

    // 이전 프레임 오차 (D 제어용)
    double prev_error_x_;
    double prev_error_y_;

    // 데드존 및 리미트 설정 범위
    double dead_zone_px_;     // 데드존 (기본값: 10 픽셀) [cite: 65]
    double max_pan_angle_;    // Pan 가동 상한 (기본값: 180도) [cite: 67]
    double min_pan_angle_;    // Pan 가동 하한 (기본값: 0도) [cite: 67]
    double max_tilt_angle_;   // Tilt 가동 상한 (기본값: 150도)
    double min_tilt_angle_;   // Tilt 가동 하한 (기본값: 30도)

    // 카메라 기준 해상도 정보
    const double center_x_ = 320.0; // 640x480 해상도의 중심 [cite: 32]
    const double center_y_ = 240.0;

    /**
     * @brief 값이 제한 범위를 벗어나지 않도록 클램핑하는 템플릿 함수
     */
    template <typename T>
    T clamp(const T& value, const T& low, const T& high) {
        return (value < low) ? low : (high < value) ? high : value;
    }
};