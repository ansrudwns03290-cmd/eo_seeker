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

        tracker.setTargetDescriptors(acq_manager.getTargetDescriptors()); 
        tracker.setTargetTemplate(roiImg);

        if (!tracker.init(current_frame.image, target_box)) {
            std::cerr << "[Error] Tracker 초기화 실패!" << std::endl;
            return -1;
        }
    
        // 시연을 위해 추출된 표적 알맹이(몽타주)를 별도 창으로 표시
        cv::imshow("Captured Target Model", roiImg);
        
        std::cout << "[Init] Target & Tracker Registered!" << std::endl;
    } else {
        std::cout << "[Cancel] ROI 선택이 취소되었습니다." << std::endl;
        return 0;
    }

    // 3. 루프 시작: 원본(컬러)과 전처리(흑백) 영상을 동시에 출력
    
    double conf_sum = 0; //NCC와 ORB 기반 신뢰도 비교를 위한 카운터 (디버그용)
    while (true) {
        if (!video_input.read(current_frame)) break;
        
        bool isFound = tracker.update(current_frame.image, target_box);
        float conf = tracker.getConfidence();
        conf_sum += conf;

        // 메타데이터 확인 로그 (Resolution, Timestamp)
        std::cout << "Frame: " << current_frame.width << "x" << current_frame.height
                  << " | Count: " << current_frame.frame_count
                  << " | Conf: " << std::fixed << std::setprecision(2) << conf
                  << " | TS: " << current_frame.timestamp_ms << "ms" << std::endl;

        // [Step A] 전처리 수행 (컬러 -> 흑백 변환)
        cv::Mat processed_img;
        if (!preprocessor.process(current_frame, processed_img)) continue;

        

        // [Step B] 시각화 준비
        cv::Mat display_img = current_frame.image.clone();
        
        if (isFound) {
            // 추적 성공 시: 초록색 사각형과 신뢰도 표시
            cv::rectangle(display_img, target_box, cv::Scalar(0, 255, 0), 2);
            cv::putText(display_img, "STATE: TRACKING", cv::Point(15, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(display_img, "Conf: " + std::to_string(conf), 
                        cv::Point(15, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        } else {
            // 추적 실패 시: 빨간색 안내문 표시
            cv::putText(display_img, "STATE: LOST", cv::Point(15, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    
        }
        
        // 프레임 정보 표시
        cv::putText(display_img, "F: " + std::to_string(current_frame.frame_count), 
                    cv::Point(current_frame.width - 100, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

        // [Step D] 화면 출력
        cv::imshow("Tracking Test", display_img);

        // ESC 종료
        if (cv::waitKey(1) == 27) break;
    }

    
    std::cout << "Average Confidence: " << (conf_sum / current_frame.frame_count) << std::endl; //디버그용 평균 신뢰도 출력
    video_input.release();
    cv::destroyAllWindows();
    return 0;
}