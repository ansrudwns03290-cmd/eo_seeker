#include <iostream>
#include <opencv2/opencv.hpp>

#include "input/VideoInput.hpp"
#include "common/Frame.hpp"
#include "preprocessing/Preprocessor.hpp"
#include "acquisition/AcquisitionManager.hpp"
#include "target_tracking/Tracker.hpp"

int main() {
    // 1. 모듈 객체 생성
    VideoInput video_input;
    Preprocessor preprocessor;
    AcquisitionManager acq_manager;
    Tracker tracker;

    // 2. 카메라 열기
    if (!video_input.open(0)) {
        std::cerr << "[Error] 카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    std::cout << "=== Acquisition & Preprocessing 통합 테스트 ===" << std::endl;
    std::cout << "1. 'Original' 창에서 표적을 드래그하세요." << std::endl;
    std::cout << "2. ENTER를 누르면 획득이 완료됩니다." << std::endl;

    Frame current_frame;
    cv::Rect target_box;

    // 1. 초기 ROI 설정을 위한 프레임 획득
    if (!video_input.read(current_frame)) return -1;

    // 2. 원본 영상 창에서 ROI 선택
    cv::namedWindow("1. Original Video", cv::WINDOW_AUTOSIZE);
    target_box = cv::selectROI("1. Original Video", current_frame.image, false);

    if (target_box.width > 0 && target_box.height > 0) {
        // 이미지 경계 안전 처리
        cv::Rect img_rect(0, 0, current_frame.width, current_frame.height);
        target_box = target_box & img_rect;

        // 표적 모델 등록 (이미지 조각 전달)
        cv::Mat roiImg = current_frame.image(target_box);
        acq_manager.setTargetModel(roiImg);

        // 시연을 위해 추출된 표적 알맹이(몽타주)를 별도 창으로 표시
        cv::imshow("Captured Target Model", roiImg);
        
        std::cout << "[Init] Target Registered!" << std::endl;
    } else {
        std::cout << "[Cancel] ROI 선택이 취소되었습니다." << std::endl;
        return 0;
    }

    // 3. 루프 시작: 원본(컬러)과 전처리(흑백) 영상을 동시에 출력
    while (true) {
        if (!video_input.read(current_frame)) break;

        // 메타데이터 확인 로그 (Resolution, Timestamp)
        std::cout << "Frame: " << current_frame.width << "x" << current_frame.height
                  << " | Count: " << current_frame.frame_count
                  << " | TS: " << current_frame.timestamp_ms << "ms" << std::endl;

        // [Step A] 전처리 수행 (컬러 -> 흑백 변환)
        cv::Mat processed_img;
        if (!preprocessor.process(current_frame, processed_img)) continue;

        // [Step B] 시각화용 이미지 준비
        // 원본 창용 (복제)
        cv::Mat color_display = current_frame.image.clone();
        // 흑백 창용 (BBox를 그리기 위해 다시 3채널 컬러 공간으로 변환하지만 색은 회색조 유지)
        cv::Mat gray_display;
        cv::cvtColor(processed_img, gray_display, cv::COLOR_GRAY2BGR);

        // [Step C] BBox 및 정보 표시 (흑백 영상 위에 표시)
        // 흑백 처리된 영상 위에 초록색 사각형 표시
        cv::rectangle(gray_display, target_box, cv::Scalar(0, 255, 0), 2);
        cv::putText(gray_display, "PROCESSING DATA (GRAY)", cv::Point(15, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // 원본 영상에는 간단한 안내 텍스트만 표시
        cv::putText(color_display, "ORIGINAL SOURCE", cv::Point(15, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        // [Step D] 두 개의 창에 각각 출력
        cv::imshow("1. Original Video", color_display);
        cv::imshow("2. Preprocessed (Target BBox)", gray_display);

        // ESC 종료
        if (cv::waitKey(30) == 27) break;
    }

    video_input.release();
    cv::destroyAllWindows();
    return 0;
}