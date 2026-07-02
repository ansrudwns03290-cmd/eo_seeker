#include "control/ControlCommand.hpp"
#include <iostream>
#include <cmath>

ControlCommand::ControlCommand(double p_gain_x, double p_gain_y, double d_gain_x, double d_gain_y)
    : kp_x_(p_gain_x), kp_y_(p_gain_y), kd_x_(d_gain_x), kd_y_(d_gain_y),
      prev_error_x_(0.0), prev_error_y_(0.0),
      dead_zone_px_(10.0), // 필수 요구사항 기준인 ±10 픽셀 설정 [cite: 65]
      min_pan_angle_(0.0), max_pan_angle_(180.0),
      min_tilt_angle_(30.0), max_tilt_angle_(150.0) {}

ServoCommand ControlCommand::calculateCommand(double target_x, double target_y, const std::string& fsm_state) {
    ServoCommand cmd{0.0, 0.0};
    static double current_pan_angle = 90.0;  // 초기 Pan 각도 (중립 위치)
    static double current_tilt_angle = 90.0; // 초기 Tilt 각도 (중립 위치)

    // 1. TRACK 상태가 아니거나 표적 유실 상황인 경우 처리
    if (fsm_state != "TRACK" && fsm_state != "REACQUIRE") {
        // SEARCH나 LOST 모드인 상태에서는 모터를 급격하게 구동하지 않고 마지막 유지 각도를 그대로 반환(정지 유지)
        prev_error_x_ = 0.0;
        prev_error_y_ = 0.0;
        return ServoCommand{current_pan_angle, current_tilt_angle};
    }

    // 2. 화면 중심과 표적 예측 위치 사이의 오차 계산 [cite: 64]
    double error_x = target_x - center_x_;
    double error_y = target_y - center_y_;

    // 3. 데드존(Dead Zone) 적용 [cite: 65]
    // 오차 크기가 지정한 데드존(10픽셀) 이내이면 서보 모터의 떨림(Jitter)을 막기 위해 오차를 0으로 강제화 [cite: 65]
    if (std::abs(error_x) <= dead_zone_px_) {
        error_x = 0.0;
    }
    if (std::abs(error_y) <= dead_zone_px_) {
        error_y = 0.0;
    }

    // 4. P/PD 제어 알고리즘 연산 수행 [cite: 64]
    double delta_error_x = error_x - prev_error_x_;
    double delta_error_y = error_y - prev_error_y_;

    double pan_output_delta = (error_x * kp_x_) + (delta_error_x * kd_x_);
    double tilt_output_delta = (error_y * kp_y_) + (delta_error_y * kd_y_);

    current_pan_angle += pan_output_delta;
    current_tilt_angle += tilt_output_delta;

    prev_error_x_ = error_x;
    prev_error_y_ = error_y;

    double raw_pan = current_pan_angle;
    double raw_tilt = current_tilt_angle;

    // 오차 보정량 계산 (화면 중심 오차를 상쇄하기 위한 변위 연산)
    cmd.pan_cmd  = clamp(current_pan_angle, min_pan_angle_, max_pan_angle_);
    cmd.tilt_cmd = clamp(current_tilt_angle, min_tilt_angle_, max_tilt_angle_);

    // 한계 도달 여부 판단
    bool pan_at_limit  = (raw_pan  != cmd.pan_cmd);
    bool tilt_at_limit = (raw_tilt != cmd.tilt_cmd);

    // 상태 변화 시에만 출력 (매 프레임 출력 방지)
    static bool prev_pan_limit  = false;
    static bool prev_tilt_limit = false;

    if (pan_at_limit && !prev_pan_limit)
        std::cout << "[Servo WARNING] Pan 한계 도달: " << cmd.pan_cmd << "°" << std::endl;
    if (!pan_at_limit && prev_pan_limit)
        std::cout << "[Servo] Pan 한계 해제" << std::endl;

    if (tilt_at_limit && !prev_tilt_limit)
        std::cout << "[Servo WARNING] Tilt 한계 도달: " << cmd.tilt_cmd << "°" << std::endl;
    if (!tilt_at_limit && prev_tilt_limit)
        std::cout << "[Servo] Tilt 한계 해제" << std::endl;

    prev_pan_limit  = pan_at_limit;
    prev_tilt_limit = tilt_at_limit;
    
    current_pan_angle = cmd.pan_cmd;
    current_tilt_angle = cmd.tilt_cmd;

    return cmd;
}

void ControlCommand::setPGains(double p_x, double p_y) {
    kp_x_ = p_x;
    kp_y_ = p_y;
}

void ControlCommand::setDGains(double d_x, double d_y) {
    kd_x_ = d_x;
    kd_y_ = d_y;
}

void ControlCommand::setDeadZone(double pixels) {
    dead_zone_px_ = pixels;
}

void ControlCommand::setOutputLimits(double pan_min, double pan_max, double tilt_min, double tilt_max) {
    min_pan_angle_ = pan_min;
    max_pan_angle_ = pan_max;
    min_tilt_angle_ = tilt_min;
    max_tilt_angle_ = tilt_max;
}