cmake_minimum_required(VERSION 3.10)
project(eo_seeker)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 패키지 찾기
find_package(OpenCV REQUIRED)

# Raspberry Pi 빌드 여부 (PCA9685 실제 PWM 출력 vs PC 텍스트 스텁)
# 사용법: cmake --preset pi-debug 또는 -DRASPBERRY_PI_BUILD=ON
option(RASPBERRY_PI_BUILD "Build with Raspberry Pi hardware output (PCA9685)" OFF)

# 실행 파일 생성
# 소스 파일이 늘어날 것에 대비해 변수로 관리하면 편합니다.
set(SOURCE_FILES
    src/main.cpp
    src/input/VideoInput.cpp
    src/acquisition/AcquisitionManager.cpp
    src/target_tracking/Tracker.cpp
    src/fsm/FsmModule.cpp
    src/control/ControlCommand.cpp
    src/preprocessing/Preprocessor.cpp
    src/state_estimation/StateEstimator.cpp
    src/hardware_output/ServoOutput.cpp
)

add_executable(eo_seeker ${SOURCE_FILES})

# 헤더 파일 경로 설정 (include 폴더)
target_include_directories(eo_seeker PRIVATE include)

# OpenCV 라이브러리 링크
target_link_libraries(eo_seeker PRIVATE ${OpenCV_LIBS})

# 빌드 시 상세 정보 출력 (선택 사항)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# ServoOutput 독립 테스트용 실행 파일 (OpenCV 의존성 없음)
add_executable(test_servo_output
    tests/test_servo_output.cpp
    src/hardware_output/ServoOutput.cpp
)
target_include_directories(test_servo_output PRIVATE include)

if(RASPBERRY_PI_BUILD)
    target_compile_definitions(eo_seeker PRIVATE RASPBERRY_PI_BUILD)
    target_compile_definitions(test_servo_output PRIVATE RASPBERRY_PI_BUILD)
endif()

# KCF 추적기 단독 검증용 실행 파일
# AcquisitionManager/StateEstimator/FSM 등 다른 로직 없이 cv::TrackerKCF만으로
# 같은 영상에서 표적을 따라가는지 확인하기 위한 독립 실행 파일 (OpenCV만 의존)
add_executable(test_kcf_standalone tests/test_kcf_standalone.cpp)
target_link_libraries(test_kcf_standalone PRIVATE ${OpenCV_LIBS})