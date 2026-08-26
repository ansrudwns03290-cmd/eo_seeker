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

add_executable(test_video_backend test_video_backend.cpp)
target_link_libraries(test_video_backend PRIVATE ${OpenCV_LIBS})
target_include_directories(test_video_backend PRIVATE ${OpenCV_INCLUDE_DIRS})

# ============================================================
# GoogleTest 도입 (검증 계획 4단계) — FetchContent로 자동 다운로드/빌드되므로
# 별도 설치 없이 빌드 시점에 바로 사용 가능.
# ============================================================
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# StateEstimator(칼만필터) 단위테스트: 이미지 의존성이 없어 GoogleTest 도입의
# 첫 검증 대상으로 삼음. 통과 확인 후 Tracker 등 다른 모듈도 같은 패턴으로 추가 예정.
add_executable(test_state_estimator
    tests/test_state_estimator.cpp
    src/state_estimation/StateEstimator.cpp
)
target_include_directories(test_state_estimator PRIVATE include)
target_link_libraries(test_state_estimator PRIVATE gtest_main ${OpenCV_LIBS})

include(GoogleTest)
gtest_discover_tests(test_state_estimator)

# Tracker(KCF) 단위테스트: 합성 이미지로 입력 검증/기본 추적 계약만 확인하고,
# ORB/NCC 신뢰도 정밀 검증이나 30프레임 주기 템플릿 갱신 같은 실카메라 튜닝
# 의존 동작은 실제 영상(test1~5.mp4) 기반 통합 실행으로 이미 검증되므로 범위에서 제외.
add_executable(test_tracker
    tests/test_tracker.cpp
    src/target_tracking/Tracker.cpp
)
target_include_directories(test_tracker PRIVATE include)
target_link_libraries(test_tracker PRIVATE gtest_main ${OpenCV_LIBS})
gtest_discover_tests(test_tracker)
