#include "hardware_output/ServoOutput.hpp"
#include "common/ServoCommand.hpp"
#include <iostream>

// ServoOutput 모듈 단독 검증용 테스트
// 카메라/트래킹/FSM 파이프라인 없이 ServoCommand 값을 직접 넣어 각도 클램핑, 중립 복귀 동작을 확인
int main() {
    std::cout << "=== ServoOutput 독립 테스트 시작 ===" << std::endl;

    // 기본 가동 범위로 생성: Pan 0~180도, Tilt 30~150도
    ServoOutput servo(0.0, 180.0, 30.0, 150.0);

    std::cout << "\n[테스트 1] 정상 범위 내 각도 (Pan 90, Tilt 90)" << std::endl;
    servo.sendCommand(ServoCommand{90.0, 90.0});

    std::cout << "\n[테스트 2] Pan 상한 초과 (Pan 200 -> 180 기대)" << std::endl;
    servo.sendCommand(ServoCommand{200.0, 90.0});

    std::cout << "\n[테스트 3] Tilt 하한 미달 (Tilt 10 -> 30 기대)" << std::endl;
    servo.sendCommand(ServoCommand{90.0, 10.0});

    std::cout << "\n[테스트 4] Pan 하한 미달 (Pan -20 -> 0 기대)" << std::endl;
    servo.sendCommand(ServoCommand{-20.0, 90.0});

    std::cout << "\n[테스트 5] 중립 복귀 (centerServos, Pan 90 / Tilt 90 기대)" << std::endl;
    servo.centerServos();

    std::cout << "\n=== 테스트 종료 ===" << std::endl;
    std::cout << "(참고: servo 객체 소멸 시 소멸자에서 centerServos()가 한 번 더 호출되어 출력이 한 줄 더 찍힙니다)" << std::endl;
    return 0;
}
