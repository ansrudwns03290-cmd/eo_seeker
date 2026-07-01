#pragma once

/**
 * @brief 서보 모터 제어 명령을 담는 구조체
 *        ControlCommand(생성) → ServoOutput(출력) 두 모듈이 공유하는 데이터 계약
 */
struct ServoCommand {
    double pan_cmd;   // Pan 서보 제어 출력값 (각도 또는 변위)
    double tilt_cmd;  // Tilt 서보 제어 출력값 (각도 또는 변위)
};
