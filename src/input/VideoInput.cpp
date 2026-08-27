#include "input/VideoInput.hpp"
#include <chrono>
#include <iostream>

VideoInput::VideoInput() : frame_count_(0) {
    // 객체 생성 시 프레임 카운트 초기화 [cite: 467]
}

VideoInput::~VideoInput() {
    release();
}

bool VideoInput::open(int camera_id) {
    // 카메라 열기 시도 [cite: 70]
#ifdef RASPBERRY_PI_BUILD
    // Pi에서 libcamera 카메라를 v4l2-compat shim(LD_PRELOAD)으로 열 때,
    // 백엔드를 지정하지 않으면 OpenCV가 GStreamer 백엔드를 우선 선택해버려서
    // 첫 프레임(ROI 선택용)은 성공하지만 두 번째 프레임부터
    // "Failed to allocate a buffer"로 실패하는 것이 Pi 실기에서 확인됨
    // (GStreamer v4l2src의 버퍼 협상 방식이 v4l2-compat shim과 호환 안 됨).
    // 순수 V4L2 ioctl 경로(cap_v4l.cpp)는 동일 조건에서 안정적으로 동작하므로
    // Pi 빌드에서는 명시적으로 V4L2 백엔드를 강제한다.
    bool camera_opened = cap_.open(camera_id, cv::CAP_V4L2);
#else
    bool camera_opened = cap_.open(camera_id);
#endif
    if (!camera_opened) {
        std::cerr << "[Error] Failed to open camera ID: " << camera_id << std::endl;
        return false;
    }

    // 프로젝트 표준 사양 강제 설정 (640x480) [cite: 69, 444]
    setResolution(640, 480);
    
    // 실시간성 확보: 오래된 프레임이 쌓이지 않도록 버퍼 1로 제한 [cite: 70, 448]
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // [진단용] 드라이버가 보고하는 이론상 FPS 확인 (실측치와 비교용, 참고 수치)
    std::cout << "[VideoInput] Driver-reported FPS: " << cap_.get(cv::CAP_PROP_FPS) << std::endl;

    return cap_.isOpened();
}

bool VideoInput::openFile(const std::string& path) {
    if (!cap_.open(path)) {
        std::cerr << "[Error] Failed to open video file: " << path << std::endl;
        return false;
    }

    m_isFileMode = true; // 재현 가능한 테스트를 위해 파일 재생 모드로 전환

    return cap_.isOpened();
}

bool VideoInput::read(Frame& frame) {
    cv::Mat img;

    // 프레임 읽기 및 유효성 검사 [cite: 71, 75]
    if (!cap_.read(img) || img.empty()) {
        frame.is_valid = false; // Frame.hpp의 멤버 변수명 반영
        return false;
    }

    // 파일 재생 모드: cap_.set(CAP_PROP_FRAME_WIDTH/HEIGHT)는 이미 인코딩된
    // 파일에는 적용되지 않으므로, 실제 카메라(640x480)와 동일한 조건으로
    // 테스트하기 위해 여기서 명시적으로 리사이즈한다.
    if (m_isFileMode && (img.cols != 640 || img.rows != 480)) {
        cv::resize(img, img, cv::Size(640, 480));
    }

    // 데이터 구조(Frame.hpp) 규격에 따른 패키징 [cite: 61, 375]
    frame.image = img;
    frame.width = img.cols;
    frame.height = img.rows;
    frame.frame_count = frame_count_++;        // 64비트 일련번호 부여

    // 파일 재생 모드에서는 처리 속도에 좌우되는 실시간 타임스탬프 대신
    // 영상 자체의 타임코드(CAP_PROP_POS_MSEC)를 사용해, 실행할 때마다
    // dt/칼만 필터 계산이 항상 동일하게 재현되도록 한다.
    frame.timestamp_ms = m_isFileMode
        ? static_cast<uint64_t>(cap_.get(cv::CAP_PROP_POS_MSEC))
        : getCurrentTimestampMs(); // 정밀 타임스탬프 기록 [cite: 443]
    frame.is_valid = true;

    return true;
}

void VideoInput::setResolution(int width, int height) {
    if (cap_.isOpened()) {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
}

void VideoInput::release() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

// 내부 보조 함수: 현재 시스템 시간을 밀리초 단위로 반환 [cite: 403, 469]
uint64_t VideoInput::getCurrentTimestampMs() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
}
