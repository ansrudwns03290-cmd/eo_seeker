#include <gtest/gtest.h>
#include "control/ControlCommand.hpp"

TEST(ControlCommandTest, SequentialBehavior) {
    ControlCommand ctrl(0.02, 0.02, 0.002, 0.002);

    // 1. 표적이 화면 중앙(320,240)에 있으면 데드존(±10px) 안이라 각도가
    //    초기값(90도)에서 그대로 유지되어야 함
    ServoCommand centered = ctrl.calculateCommand(320.0, 240.0, "TRACK");
    EXPECT_NEAR(centered.pan_cmd, 90.0, 1e-6);
    EXPECT_NEAR(centered.tilt_cmd, 90.0, 1e-6);

    // 2. 표적이 화면 중앙보다 오른쪽(x가 큼)에 있으면 Pan 각도가 증가해야 함
    ServoCommand rightOffset = ctrl.calculateCommand(420.0, 240.0, "TRACK");
    EXPECT_GT(rightOffset.pan_cmd, centered.pan_cmd);

    // 3. TRACK/REACQUIRE가 아닌 상태(SEARCH, LOST 등)에서는 모터를 급격히
    //    구동하지 않고 방금 각도를 그대로 유지해야 함
    ServoCommand held = ctrl.calculateCommand(0.0, 0.0, "LOST");
    EXPECT_NEAR(held.pan_cmd, rightOffset.pan_cmd, 1e-6);
    EXPECT_NEAR(held.tilt_cmd, rightOffset.tilt_cmd, 1e-6);

    // 4. 극단적으로 큰 오차를 반복해서 넣으면 Pan 각도가 상한(180도)에서
    //    clamp되어야 함
    ServoCommand clamped{0.0, 0.0};
    for (int i = 0; i < 50; ++i) {
        clamped = ctrl.calculateCommand(10000.0, 240.0, "TRACK");
    }
    EXPECT_NEAR(clamped.pan_cmd, 180.0, 1e-6);
}
