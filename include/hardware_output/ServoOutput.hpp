#pragma once

#include "common/ServoCommand.hpp"

/**
 * @brief 서보 한 축의 하드웨어 캘리브레이션 설정 (플랫폼 공통, 하드웨어 접근 없음)
 *
 * 각도(도) 지령은 "90도 = 중립(center_tick)" 기준 절대각이다 (ControlCommand의 중립 90도와 동일).
 * 기계적 한계는 각도가 아니라 tick으로 정의한다 — deg/tick 비율 오차가 한계 보호에 섞이지 않게 하기 위함.
 */
struct AxisConfig {
    int    channel;        // PCA9685 채널 번호 (0~15)
    int    center_tick;    // 중립(정면/수평) tick — 90도 지령에 대응
    int    min_tick;       // 출력 허용 하한 tick (기계 한계에서 여유를 둔 값)
    int    max_tick;       // 출력 허용 상한 tick (기계 한계에서 여유를 둔 값)
    bool   invert;         // true: tick 감소 = 각도 증가(+), false: tick 증가 = 각도 증가(+)
    double ticks_per_deg;  // 1도당 tick (> 0). 실측 전에는 이론값 사용: 2000us/180도 -> 약 2.275
};

/**
 * @brief ServoCommand를 실제 하드웨어(PCA9685 PWM) 또는 PC 콘솔 스텁으로 출력하는 클래스
 *        각도 -> tick 변환/클램프/한계각 계산은 플랫폼 공통 순수 로직이라 PC에서도 단위테스트 가능하고,
 *        Pi 빌드(RASPBERRY_PI_BUILD)에서만 raw Linux i2c-dev로 PCA9685에 실제로 쓴다.
 *
 * 각도 부호 규칙 (ControlCommand와 동일):
 *   Pan : +각도 = 오른쪽,  Tilt : +각도 = 아래쪽 (영상 y축이 아래로 증가하므로)
 */
class ServoOutput {
public:
    static constexpr double kNeutralAngleDeg = 90.0;

    /// 2026-10 캘리브레이션 값 (Pan: 정면 322, 144~500 대칭 / Tilt: 수평 367, 217~417)
    static AxisConfig defaultPanConfig();
    static AxisConfig defaultTiltConfig();

    /**
     * @brief 하드웨어 출력 모듈 생성자
     *        Pi 빌드에서는 PCA9685를 50Hz로 초기화하고 서보를 중립으로 이동시킨다.
     *        I2C 초기화에 실패하면 오류를 출력하고 서보 출력 없이 계속 동작한다.
     */
    explicit ServoOutput(const AxisConfig& pan = defaultPanConfig(),
                         const AxisConfig& tilt = defaultTiltConfig());

    /// 소멸 시 서보를 중립으로 되돌리고 (Pi 빌드) 이동 시간 대기 후 PWM 출력을 끈다.
    ~ServoOutput();

    ServoOutput(const ServoOutput&) = delete;
    ServoOutput& operator=(const ServoOutput&) = delete;

    /**
     * @brief ServoCommand(pan_cmd, tilt_cmd: 90도 중심 절대각)를 받아 서보로 출력
     *        각도 -> tick 변환 후 tick 한계로 클램프한다. 직전과 같은 tick이면 I2C 쓰기를 생략한다.
     */
    void sendCommand(const ServoCommand& cmd);

    /// 서보를 중립(center_tick)으로 되돌린다 (프로그램 종료/예외 상황용)
    void centerServos();

    // --- tick 한계에서 역산한 실제 가동 각도 범위 (ControlCommand::setOutputLimits에 주입용) ---
    double panMinAngle()  const { return minAngle(pan_); }
    double panMaxAngle()  const { return maxAngle(pan_); }
    double tiltMinAngle() const { return minAngle(tilt_); }
    double tiltMaxAngle() const { return maxAngle(tilt_); }

    /// 마지막으로 출력(성공)된 tick. 아직 출력 전이면 -1.
    int lastPanTick()  const { return last_pan_tick_; }
    int lastTiltTick() const { return last_tilt_tick_; }

    // --- 순수 변환 로직 (하드웨어 접근 없음, 단위테스트 대상) ---
    /// 각도(90도 중심) -> tick. 결과는 [min_tick, max_tick]으로 클램프된다.
    static int    angleToTick(const AxisConfig& axis, double angle_deg);
    /// tick -> 각도(90도 중심). 클램프하지 않는다.
    static double tickToAngle(const AxisConfig& axis, int tick);
    /// tick 한계(min/max)에 대응하는 각도 범위
    static double minAngle(const AxisConfig& axis);
    static double maxAngle(const AxisConfig& axis);

private:
    AxisConfig pan_;
    AxisConfig tilt_;
    int last_pan_tick_  = -1;
    int last_tilt_tick_ = -1;

    /// 계산된 tick을 출력하고 last_*_tick_을 갱신한다 (Pi 빌드: 변경된 축만 I2C로 기록)
    void outputTicks(int pan_tick, int tilt_tick);

#ifdef RASPBERRY_PI_BUILD
    int  i2c_fd_ = -1;                  // /dev/i2c-1 파일 디스크립터 (-1: 미연결/초기화 실패)
    bool write_error_reported_ = false; // I2C 쓰기 오류는 프레임마다 반복 출력하지 않고 1회만 보고

    bool initPca9685();                         // 50Hz 설정 (SLEEP -> PRESCALE -> wake -> RESTART)
    bool writeReg(unsigned char reg, unsigned char value);
    bool writeTick(int channel, int tick);      // ON=0, OFF=tick 을 5바이트 블록으로 기록 (Auto-Increment)
#endif
};
