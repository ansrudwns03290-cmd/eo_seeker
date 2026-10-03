#include "hardware_output/ServoOutput.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

#ifdef RASPBERRY_PI_BUILD
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#ifdef RASPBERRY_PI_BUILD
namespace {
// --- PCA9685 (datasheet 기준) ---
constexpr const char*   kI2cDevice       = "/dev/i2c-1";   // Pi의 GPIO2(SDA)/GPIO3(SCL)
constexpr int           kPcaAddress      = 0x40;           // 기본 주소 (i2cdetect로 확인됨)
constexpr unsigned char kRegMode1        = 0x00;
constexpr unsigned char kRegPrescale     = 0xFE;
constexpr unsigned char kRegLed0OnL      = 0x06;           // 채널 n: 0x06 + 4*n (ON_L, ON_H, OFF_L, OFF_H)
constexpr unsigned char kMode1Sleep      = 0x10;           // PRESCALE은 SLEEP 상태에서만 기록된다
constexpr unsigned char kMode1AutoInc    = 0x20;           // 블록 쓰기용 레지스터 주소 자동 증가
constexpr unsigned char kMode1Restart    = 0x80;           // SLEEP 이후 PWM 재시작
constexpr unsigned char kPrescale50Hz    = 0x79;           // round(25MHz / (4096 * 50)) - 1 = 121
constexpr unsigned char kFullOffBit      = 0x10;           // LEDn_OFF_H bit4: 해당 채널 출력 완전 OFF
constexpr int           kStartupSettleUs = 5000;           // wake 이후 오실레이터 안정화 대기
constexpr int           kShutdownMoveUs  = 300000;         // 종료 시 중립 복귀 이동 시간 확보 후 출력 OFF
}  // namespace
#endif

AxisConfig ServoOutput::defaultPanConfig() {
    // 정면 322(눈대중), 한계 144~500(중앙 기준 대칭), tick 감소 = 오른쪽(+) -> invert
    return AxisConfig{0, 322, 144, 500, true, 2.275};
}

AxisConfig ServoOutput::defaultTiltConfig() {
    // 수평 367, 한계 217~417(위쪽 150 / 아래쪽 50 tick: 표적이 위쪽인 시나리오 위주),
    // tick 증가 = 아래쪽(+) -> invert 아님
    return AxisConfig{1, 367, 217, 417, false, 2.275};
}

int ServoOutput::angleToTick(const AxisConfig& axis, double angle_deg) {
    // 비정상 입력(NaN)이 서보를 임의 위치로 보내지 않도록 중립으로 처리한다.
    if (std::isnan(angle_deg)) return axis.center_tick;
    const double sign = axis.invert ? -1.0 : 1.0;
    const double raw  = axis.center_tick + sign * (angle_deg - kNeutralAngleDeg) * axis.ticks_per_deg;
    // 큰 값이 정수 변환에서 넘치지 않도록 실수 상태에서 먼저 클램프한 뒤 반올림한다.
    const double clamped = std::clamp(raw, static_cast<double>(axis.min_tick),
                                           static_cast<double>(axis.max_tick));
    return static_cast<int>(std::lround(clamped));
}

double ServoOutput::tickToAngle(const AxisConfig& axis, int tick) {
    const double sign = axis.invert ? -1.0 : 1.0;
    return kNeutralAngleDeg + sign * (tick - axis.center_tick) / axis.ticks_per_deg;
}

double ServoOutput::minAngle(const AxisConfig& axis) {
    return std::min(tickToAngle(axis, axis.min_tick), tickToAngle(axis, axis.max_tick));
}

double ServoOutput::maxAngle(const AxisConfig& axis) {
    return std::max(tickToAngle(axis, axis.min_tick), tickToAngle(axis, axis.max_tick));
}

ServoOutput::ServoOutput(const AxisConfig& pan, const AxisConfig& tilt)
    : pan_(pan), tilt_(tilt) {
    char range_msg[160];
    std::snprintf(range_msg, sizeof(range_msg),
                  "Pan %.1f~%.1f도 / Tilt %.1f~%.1f도 (중립 %.0f도)",
                  panMinAngle(), panMaxAngle(), tiltMinAngle(), tiltMaxAngle(), kNeutralAngleDeg);

#ifdef RASPBERRY_PI_BUILD
    if (initPca9685()) {
        std::cout << "[ServoOutput] PCA9685 초기화 완료 (0x40, 50Hz) - 가동 범위 " << range_msg << std::endl;
        centerServos();  // 시작 시 중립 위치로 이동
    } else {
        std::cerr << "[ServoOutput][ERROR] PCA9685 초기화 실패 - 서보 출력 없이 계속 진행합니다." << std::endl;
    }
#else
    std::cout << "[ServoOutput] PC 빌드 - 텍스트 출력 스텁 모드로 동작 - 가동 범위 " << range_msg << std::endl;
#endif
}

ServoOutput::~ServoOutput() {
    centerServos();
#ifdef RASPBERRY_PI_BUILD
    if (i2c_fd_ >= 0) {
        // 중립 복귀 명령 직후 바로 끄면 이동 도중에 힘이 풀리므로, 이동 시간을 확보한 뒤 PWM 출력을 끈다.
        usleep(kShutdownMoveUs);
        writeReg(static_cast<unsigned char>(kRegLed0OnL + 4 * pan_.channel + 3), kFullOffBit);
        writeReg(static_cast<unsigned char>(kRegLed0OnL + 4 * tilt_.channel + 3), kFullOffBit);
        close(i2c_fd_);
        i2c_fd_ = -1;
    }
#endif
}

void ServoOutput::sendCommand(const ServoCommand& cmd) {
    outputTicks(angleToTick(pan_, cmd.pan_cmd), angleToTick(tilt_, cmd.tilt_cmd));
}

void ServoOutput::centerServos() {
    outputTicks(pan_.center_tick, tilt_.center_tick);
}

void ServoOutput::outputTicks(int pan_tick, int tilt_tick) {
#ifdef RASPBERRY_PI_BUILD
    if (i2c_fd_ >= 0) {
        // 변경된 축만 기록한다. 쓰기에 실패하면 last_*_tick_을 갱신하지 않아 다음 프레임에 재시도된다.
        if (pan_tick != last_pan_tick_ && writeTick(pan_.channel, pan_tick)) {
            last_pan_tick_ = pan_tick;
        }
        if (tilt_tick != last_tilt_tick_ && writeTick(tilt_.channel, tilt_tick)) {
            last_tilt_tick_ = tilt_tick;
        }
        return;
    }
#endif
    last_pan_tick_  = pan_tick;
    last_tilt_tick_ = tilt_tick;
}

#ifdef RASPBERRY_PI_BUILD
bool ServoOutput::initPca9685() {
    i2c_fd_ = open(kI2cDevice, O_RDWR);
    if (i2c_fd_ < 0) {
        std::cerr << "[ServoOutput][ERROR] I2C 버스를 열 수 없습니다: " << kI2cDevice
                  << " (raspi-config에서 I2C 활성화, 사용자의 i2c 그룹 권한 확인)" << std::endl;
        return false;
    }
    if (ioctl(i2c_fd_, I2C_SLAVE, kPcaAddress) < 0) {
        std::cerr << "[ServoOutput][ERROR] I2C 슬레이브 주소(0x40) 설정 실패" << std::endl;
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }

    // i2cset으로 검증한 순서 그대로: SLEEP -> PRESCALE(50Hz) -> wake + Auto-Increment -> 안정화 대기 -> RESTART
    const bool ok = writeReg(kRegMode1, kMode1Sleep)
                 && writeReg(kRegPrescale, kPrescale50Hz)
                 && writeReg(kRegMode1, kMode1AutoInc)
                 && (usleep(kStartupSettleUs) == 0)
                 && writeReg(kRegMode1, static_cast<unsigned char>(kMode1Restart | kMode1AutoInc));
    if (!ok) {
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }
    return true;
}

bool ServoOutput::writeReg(unsigned char reg, unsigned char value) {
    const unsigned char buf[2] = {reg, value};
    if (write(i2c_fd_, buf, 2) == 2) return true;
    if (!write_error_reported_) {
        write_error_reported_ = true;
        std::cerr << "[ServoOutput][ERROR] PCA9685 레지스터 쓰기 실패 (reg 0x" << std::hex << int(reg)
                  << std::dec << ") - 배선/전원을 확인하세요. 이후 동일 오류는 출력하지 않습니다." << std::endl;
    }
    return false;
}

bool ServoOutput::writeTick(int channel, int tick) {
    // ON = 0, OFF = tick. OFF_H를 통째로 덮어쓰므로 이전의 full-off 비트(종료 시 설정)도 함께 해제된다.
    const unsigned char buf[5] = {
        static_cast<unsigned char>(kRegLed0OnL + 4 * channel),
        0x00, 0x00,
        static_cast<unsigned char>(tick & 0xFF),
        static_cast<unsigned char>((tick >> 8) & 0x0F)};
    if (write(i2c_fd_, buf, 5) == 5) return true;
    if (!write_error_reported_) {
        write_error_reported_ = true;
        std::cerr << "[ServoOutput][ERROR] PCA9685 채널 " << channel
                  << " 쓰기 실패 - 배선/전원을 확인하세요. 이후 동일 오류는 출력하지 않습니다." << std::endl;
    }
    return false;
}
#endif
