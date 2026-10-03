#include "hardware_output/ServoOutput.hpp"
#include "common/ServoCommand.hpp"
#include <iostream>

#ifdef RASPBERRY_PI_BUILD
#include <chrono>
#include <thread>
#endif

// ServoOutput 모듈 단독 검증용 수동 스모크 테스트 (자동 판정 없음 — 출력과 실제 서보 동작을 눈으로 확인)
// - PC 빌드 : 각도 -> tick 변환/클램프 결과를 콘솔로 확인 (자동 검증은 test_servo_axis)
// - Pi 빌드(RASPBERRY_PI_BUILD): 실제 PCA9685로 출력되어 서보가 움직인다. 각 단계 사이에 1초 대기.
//   한계 지령 단계에서는 서보가 캘리브레이션된 tick 한계 위치까지 움직이므로, 케이블/기구 간섭을 지켜볼 것.
static void step(ServoOutput& servo, const char* title, double pan_deg, double tilt_deg) {
    std::cout << "\n" << title << std::endl;
    servo.sendCommand(ServoCommand{pan_deg, tilt_deg});
    std::cout << "  -> pan tick=" << servo.lastPanTick() << ", tilt tick=" << servo.lastTiltTick() << std::endl;
#ifdef RASPBERRY_PI_BUILD
    std::this_thread::sleep_for(std::chrono::seconds(1));
#endif
}

int main() {
    std::cout << "=== ServoOutput 독립 테스트 시작 ===" << std::endl;

    // 캘리브레이션된 기본 설정으로 생성 (Pan 정면 322 / Tilt 수평 367)
    ServoOutput servo;

    std::cout << "가동 각도 범위: Pan " << servo.panMinAngle() << "~" << servo.panMaxAngle()
              << "도, Tilt " << servo.tiltMinAngle() << "~" << servo.tiltMaxAngle() << "도" << std::endl;

    step(servo, "[테스트 1] 중립 (Pan 90, Tilt 90 -> tick 322 / 367 기대)", 90.0, 90.0);
    step(servo, "[테스트 2] Pan 상한 초과 (Pan 200 -> 오른쪽 한계, tick 144 기대)", 200.0, 90.0);
    step(servo, "[테스트 3] Pan 하한 미달 (Pan -20 -> 왼쪽 한계, tick 500 기대)", -20.0, 90.0);
    step(servo, "[테스트 4] Tilt 하한 미달 (Tilt -20 -> 위쪽 한계, tick 217 기대)", 90.0, -20.0);
    step(servo, "[테스트 5] Tilt 상한 초과 (Tilt 200 -> 아래쪽 한계, tick 417 기대)", 90.0, 200.0);

    std::cout << "\n[테스트 6] 중립 복귀 (centerServos, tick 322 / 367 기대)" << std::endl;
    servo.centerServos();
    std::cout << "  -> pan tick=" << servo.lastPanTick() << ", tilt tick=" << servo.lastTiltTick() << std::endl;

    std::cout << "\n=== 테스트 종료 ===" << std::endl;
    std::cout << "(참고: servo 객체 소멸 시 중립 복귀 후 PWM 출력이 꺼진다)" << std::endl;
    return 0;
}
