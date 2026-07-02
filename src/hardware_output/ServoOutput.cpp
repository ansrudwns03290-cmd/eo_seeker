#include "hardware_output/ServoOutput.hpp"
#include <iostream>

ServoOutput::ServoOutput(double pan_min_angle, double pan_max_angle,
                          double tilt_min_angle, double tilt_max_angle)
    : pan_min_angle_(pan_min_angle), pan_max_angle_(pan_max_angle),
      tilt_min_angle_(tilt_min_angle), tilt_max_angle_(tilt_max_angle) {
#ifdef RASPBERRY_PI_BUILD
    pan_channel_  = 0;  // PCA9685 채널 0번에 Pan 서보 연결
    tilt_channel_ = 1;  // PCA9685 채널 1번에 Tilt 서보 연결
    // TODO: PCA9685 I2C 디바이스 오픈 및 초기화 (주소 0x40, PWM 주파수 50Hz)
    pca9685_handle_ = -1;
    std::cout << "[ServoOutput] PCA9685 초기화 (Pi 빌드)" << std::endl;
#else
    std::cout << "[ServoOutput] PC 빌드 - 텍스트 출력 스텁 모드로 동작" << std::endl;
#endif
}

ServoOutput::~ServoOutput() {
    centerServos();
#ifdef RASPBERRY_PI_BUILD
    // TODO: PCA9685 I2C 디바이스 핸들 해제
#endif
}

void ServoOutput::sendCommand(const ServoCommand& cmd) {
    double pan_angle  = clampAngle(cmd.pan_cmd,  pan_min_angle_,  pan_max_angle_);
    double tilt_angle = clampAngle(cmd.tilt_cmd, tilt_min_angle_, tilt_max_angle_);

#ifdef RASPBERRY_PI_BUILD
    int pan_pwm  = angleToPwm(pan_angle);
    int tilt_pwm = angleToPwm(tilt_angle);
    // TODO: PCA9685 레지스터에 pan_pwm / tilt_pwm 값을 실제 I2C write로 기록
    (void)pan_pwm;
    (void)tilt_pwm;
#else
//    std::cout << "[ServoOutput] Pan: " << pan_angle << "°"
//              << " / Tilt: " << tilt_angle << "°" << std::endl;
#endif
}

void ServoOutput::centerServos() {
    ServoCommand center_cmd;
    center_cmd.pan_cmd  = (pan_min_angle_ + pan_max_angle_) / 2.0;
    center_cmd.tilt_cmd = (tilt_min_angle_ + tilt_max_angle_) / 2.0;
    sendCommand(center_cmd);
}

double ServoOutput::clampAngle(double angle, double min_angle, double max_angle) const {
    return (angle < min_angle) ? min_angle : (angle > max_angle) ? max_angle : angle;
}

#ifdef RASPBERRY_PI_BUILD
int ServoOutput::angleToPwm(double angle_deg) const {
    // MG90S 기준 펄스폭 범위: 약 500us(0도) ~ 2500us(180도), 50Hz PWM(주기 20ms), 12bit(4096) 분해능
    const double min_pulse_us = 500.0;
    const double max_pulse_us = 2500.0;
    const double period_us    = 20000.0; // 50Hz

    double pulse_us = min_pulse_us + (angle_deg / 180.0) * (max_pulse_us - min_pulse_us);
    int pwm_value = static_cast<int>((pulse_us / period_us) * 4096.0);
    return pwm_value;
}
#endif
