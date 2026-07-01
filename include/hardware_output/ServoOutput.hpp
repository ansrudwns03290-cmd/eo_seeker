#pragma once

#include "common/ServoCommand.hpp"

/**
 * @brief ServoCommand를 실제 하드웨어(PCA9685 PWM) 또는 PC 콘솔 텍스트로 출력하는 클래스
 *        PC 빌드에서는 서보 각도 값을 텍스트로 출력(스텁), Pi 빌드에서는 PCA9685를 통해 실제 PWM 신호를 출력
 */
class ServoOutput {
public:
    /**
     * @brief 하드웨어 출력 모듈 생성자
     * @param pan_min_angle Pan 서보 가동 하한(도)
     * @param pan_max_angle Pan 서보 가동 상한(도)
     * @param tilt_min_angle Tilt 서보 가동 하한(도)
     * @param tilt_max_angle Tilt 서보 가동 상한(도)
     */
    ServoOutput(double pan_min_angle = 0.0, double pan_max_angle = 180.0,
                double tilt_min_angle = 30.0, double tilt_max_angle = 150.0);

    ~ServoOutput();

    /**
     * @brief ServoCommand(pan_cmd, tilt_cmd)를 받아 실제 서보로 출력하는 핵심 함수
     * @param cmd ControlCommand가 생성한 제어 명령
     */
    void sendCommand(const ServoCommand& cmd);

    /**
     * @brief 서보를 안전한 중립 각도로 되돌리는 함수 (프로그램 종료/예외 상황용)
     */
    void centerServos();

private:
    double pan_min_angle_;
    double pan_max_angle_;
    double tilt_min_angle_;
    double tilt_max_angle_;

#ifdef RASPBERRY_PI_BUILD
    int pca9685_handle_;   // PCA9685 I2C 디바이스 핸들
    int pan_channel_;      // Pan 서보가 연결된 PCA9685 채널 번호
    int tilt_channel_;     // Tilt 서보가 연결된 PCA9685 채널 번호

    /**
     * @brief 서보 각도(도)를 PCA9685 PWM 값으로 변환
     */
    int angleToPwm(double angle_deg) const;
#endif

    /**
     * @brief 각도가 가동 범위를 벗어나지 않도록 클램핑
     */
    double clampAngle(double angle, double min_angle, double max_angle) const;
};
