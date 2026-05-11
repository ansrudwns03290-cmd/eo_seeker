#pragma once

#include <opencv2/opencv.hpp>
#include <cstdint>

struct Frame {
    cv::Mat image;              // 실제 영상 프레임
    int width = 0;              // 프레임 가로 크기
    int height = 0;             // 프레임 세로 크기
    int64_t timestamp_ms = 0;   // 프레임을 읽은 시각(ms)
    int64_t frame_count = 0;    // 프레임의 순차적 일련번호
    bool is_valid = false;      // 정상 프레임 여부
};