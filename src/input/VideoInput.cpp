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
    if (!cap_.open(camera_id)) {
        std::cerr << "[Error] Failed to open camera ID: " << camera_id << std::endl;
        return false;
    }

    // 프로젝트 표준 사양 강제 설정 (640x480) [cite: 69, 444]
    setResolution(640, 480);
    
    // 실시간성 확보: 오래된 프레임이 쌓이지 않도록 버퍼 1로 제한 [cite: 70, 448]
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);

    return cap_.isOpened();
}

bool VideoInput::openFile(const std::string& path) {
    if (!cap_.open(path)) {
        std::cerr << "[Error] Failed to open video file: " << path << std::endl;
        return false;
    }
    return cap_.isOpened();
}

bool VideoInput::read(Frame& frame) {
    cv::Mat img;

    // 프레임 읽기 및 유효성 검사 [cite: 71, 75]
    if (!cap_.read(img) || img.empty()) {
        frame.is_valid = false; // Frame.hpp의 멤버 변수명 반영
        return false;
    }

    // 데이터 구조(Frame.hpp) 규격에 따른 패키징 [cite: 61, 375]
    frame.image = img;
    frame.width = img.cols;
    frame.height = img.rows;
    frame.frame_count = frame_count_++;        // 64비트 일련번호 부여 
    frame.timestamp_ms = getCurrentTimestampMs(); // 정밀 타임스탬프 기록 [cite: 443]
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