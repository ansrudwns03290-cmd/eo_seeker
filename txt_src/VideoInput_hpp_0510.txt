#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <cstdint>
#include "common/Frame.hpp" // [cite: 455] 데이터 규격 포함

/**
 * @brief 영상 입력 및 프레임 관리를 담당하는 클래스
 */
class VideoInput {
public:
    VideoInput();
    ~VideoInput();

    // 영상 소스 연결 (성공 시 true 반환) [cite: 438, 439]
    bool open(int camera_id = 0);
    bool openFile(const std::string& path);
    
    // 핵심 기능: Frame 객체로 데이터 획득 [cite: 442, 457]
    bool read(Frame& frame);
    
    // 자원 해제 및 설정 [cite: 437, 444]
    void release();
    void setResolution(int width = 640, int height = 480);

private:
    cv::VideoCapture cap_;      // OpenCV 캡처 객체
    uint64_t frame_count_;      // 프레임 번호 카운트 [cite: 446, 458]
    
    // 타임스탬프 획득용 보조 함수 [cite: 443]
    uint64_t getCurrentTimestampMs();
};