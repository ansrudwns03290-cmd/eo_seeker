#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "hardware_output/ServoOutput.hpp"

// ServoOutput 순수 로직(각도 <-> tick 변환, 클램프, 방향/invert, 한계각 계산) 단위테스트.
// 하드웨어 접근이 없어 PC/Pi 어디서나 실행된다. 단, Pi 빌드(RASPBERRY_PI_BUILD)에서는 ServoOutput
// 객체를 만들면 실제 I2C로 서보가 구동되므로 객체 기반 테스트는 PC 빌드에서만 컴파일한다.

namespace {
const AxisConfig kPan  = ServoOutput::defaultPanConfig();
const AxisConfig kTilt = ServoOutput::defaultTiltConfig();
}  // namespace

TEST(ServoAxisTest, NeutralAngleMapsToCenterTick) {
    EXPECT_EQ(ServoOutput::angleToTick(kPan, 90.0), 322);
    EXPECT_EQ(ServoOutput::angleToTick(kTilt, 90.0), 367);
}

TEST(ServoAxisTest, DirectionFollowsInvertFlag) {
    // Pan: +각도 = 오른쪽 = tick 감소 (invert=true). 10도 * 2.275 = 22.75 tick
    EXPECT_EQ(ServoOutput::angleToTick(kPan, 100.0), 299);
    EXPECT_EQ(ServoOutput::angleToTick(kPan, 80.0), 345);
    // Tilt: +각도 = 아래쪽 = tick 증가 (invert=false)
    EXPECT_EQ(ServoOutput::angleToTick(kTilt, 100.0), 390);
    EXPECT_EQ(ServoOutput::angleToTick(kTilt, 80.0), 344);
}

TEST(ServoAxisTest, ClampsToTickLimits) {
    EXPECT_EQ(ServoOutput::angleToTick(kPan, 1000.0), kPan.min_tick);    // 144
    EXPECT_EQ(ServoOutput::angleToTick(kPan, -1000.0), kPan.max_tick);   // 500
    EXPECT_EQ(ServoOutput::angleToTick(kTilt, 1000.0), kTilt.max_tick);  // 417
    EXPECT_EQ(ServoOutput::angleToTick(kTilt, -1000.0), kTilt.min_tick); // 217

    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_EQ(ServoOutput::angleToTick(kPan, inf), kPan.min_tick);
    EXPECT_EQ(ServoOutput::angleToTick(kTilt, -inf), kTilt.min_tick);
}

TEST(ServoAxisTest, NanAngleFallsBackToCenter) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(ServoOutput::angleToTick(kPan, nan), kPan.center_tick);
    EXPECT_EQ(ServoOutput::angleToTick(kTilt, nan), kTilt.center_tick);
}

TEST(ServoAxisTest, AngleLimitsAreDerivedFromTickLimits) {
    // 2.275 tick/도 기준 (실측 후 ticks_per_deg가 바뀌면 이 기대값도 함께 갱신할 것)
    EXPECT_NEAR(ServoOutput::minAngle(kPan), 11.76, 0.05);
    EXPECT_NEAR(ServoOutput::maxAngle(kPan), 168.24, 0.05);
    EXPECT_NEAR(ServoOutput::minAngle(kTilt), 24.07, 0.05);
    EXPECT_NEAR(ServoOutput::maxAngle(kTilt), 111.98, 0.05);
}

TEST(ServoAxisTest, LimitAnglesMapBackToLimitTicks) {
    // Pan(invert): 최대각 = min_tick, 최소각 = max_tick
    EXPECT_EQ(ServoOutput::angleToTick(kPan, ServoOutput::maxAngle(kPan)), kPan.min_tick);
    EXPECT_EQ(ServoOutput::angleToTick(kPan, ServoOutput::minAngle(kPan)), kPan.max_tick);
    // Tilt(비invert): 최대각 = max_tick, 최소각 = min_tick
    EXPECT_EQ(ServoOutput::angleToTick(kTilt, ServoOutput::maxAngle(kTilt)), kTilt.max_tick);
    EXPECT_EQ(ServoOutput::angleToTick(kTilt, ServoOutput::minAngle(kTilt)), kTilt.min_tick);
}

TEST(ServoAxisTest, RoundTripWithinOneTickResolution) {
    for (const AxisConfig* axis : {&kPan, &kTilt}) {
        const double lo = ServoOutput::minAngle(*axis);
        const double hi = ServoOutput::maxAngle(*axis);
        for (double a = lo; a <= hi; a += 0.7) {
            const int tick = ServoOutput::angleToTick(*axis, a);
            EXPECT_NEAR(ServoOutput::tickToAngle(*axis, tick), a, 0.5 / axis->ticks_per_deg + 1e-9);
        }
    }
}

#ifndef RASPBERRY_PI_BUILD
TEST(ServoOutputTest, NoTicksBeforeFirstCommand) {
    ServoOutput servo;
    EXPECT_EQ(servo.lastPanTick(), -1);
    EXPECT_EQ(servo.lastTiltTick(), -1);
}

TEST(ServoOutputTest, SendCommandClampsAndRecordsTicks) {
    ServoOutput servo;
    servo.sendCommand(ServoCommand{200.0, 90.0});    // Pan 한계 초과 -> min_tick
    EXPECT_EQ(servo.lastPanTick(), 144);
    EXPECT_EQ(servo.lastTiltTick(), 367);

    servo.sendCommand(ServoCommand{90.0, -20.0});    // Tilt 위쪽 한계 초과 -> min_tick
    EXPECT_EQ(servo.lastPanTick(), 322);
    EXPECT_EQ(servo.lastTiltTick(), 217);
}

TEST(ServoOutputTest, CenterServosReturnsToCenterTicks) {
    ServoOutput servo;
    servo.sendCommand(ServoCommand{150.0, 110.0});
    servo.centerServos();
    EXPECT_EQ(servo.lastPanTick(), 322);
    EXPECT_EQ(servo.lastTiltTick(), 367);
}

TEST(ServoOutputTest, LimitAccessorsMatchAxisMath) {
    ServoOutput servo;
    EXPECT_DOUBLE_EQ(servo.panMinAngle(), ServoOutput::minAngle(kPan));
    EXPECT_DOUBLE_EQ(servo.panMaxAngle(), ServoOutput::maxAngle(kPan));
    EXPECT_DOUBLE_EQ(servo.tiltMinAngle(), ServoOutput::minAngle(kTilt));
    EXPECT_DOUBLE_EQ(servo.tiltMaxAngle(), ServoOutput::maxAngle(kTilt));
}
#endif
