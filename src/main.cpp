#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "input/VideoInput.hpp"
#include "common/Frame.hpp"
#include "preprocessing/Preprocessor.hpp"
#include "acquisition/AcquisitionManager.hpp"
#include "target_tracking/Tracker.hpp"
#include "state_estimation/StateEstimator.hpp"


int main() {
    // 1. 모듈 객체 생성
    VideoInput video_input;
    Preprocessor preprocessor;
    AcquisitionManager acq_manager;
    Tracker tracker;
    StateEstimator state_estimator;

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
    double last_timestamp_ms = 0.0;
    bool is_first_track = true;
    double conf_sum = 0; //NCC와 ORB 기반 신뢰도 비교를 위한 카운터 (디버그용)

    while (true) {
        if (!video_input.read(current_frame)) break;
        
        // 프레임 간 시간 간격(dt) 계산
        // 프레임 메타데이터의 구조적 timstap_ms를 초(seconds) 단위로 변환
        if (last_timestamp_ms == 0.0) {
            last_timestamp_ms = current_frame.timestamp_ms;
        }
        double dt = (current_frame.timestamp_ms - last_timestamp_ms) / 1000.0; // ms -> s
        last_timestamp_ms = current_frame.timestamp_ms;

        if (dt <= 0.0) dt = 0.033; // 프레임 간격이 비정상적으로 짧거나 없는 경우 기본값(30fps) 사용

        // [Step 2] 칼만 필터 예측 단계 선행 (매 프레임 무조건 먼저 수행)
        cv::Point2f predicted_pos = state_estimator.predict(dt);

        // 첫 프레임이 아니고, 칼만 필터가 이미 정상 동작 중(Initialized)일 때만 관성 유도를 적용합니다.
        if (!is_first_track && state_estimator.isInitialized()) {
            // KCF를 구동하기 전에, target_box의 중심점을 칼만이 예측한 물리적 위치로 강제 이동시킵니다.
            target_box.x = predicted_pos.x - target_box.width / 2.0f;
            target_box.y = predicted_pos.y - target_box.height / 2.0f;

            // 이미지 경계 안전 처리 (화면 밖으로 나가는 것 방지)
            cv::Rect img_rect(0, 0, current_frame.width, current_frame.height);
            target_box = target_box & img_rect;
        }

        bool isFound = tracker.update(current_frame.image, target_box);
        float conf = tracker.getConfidence();

        // 메타데이터 확인 로그 (Resolution, Timestamp)
        std::cout << "Frame: " << current_frame.width << "x" << current_frame.height
                  << " | Count: " << current_frame.frame_count
                  << " | Conf: " << std::fixed << std::setprecision(2) << conf
                  << " | TS: " << current_frame.timestamp_ms << "ms" << std::endl;

        // [디버그용 측정 오차 계산]
        if (isFound && state_estimator.isInitialized()) {
            cv::Point2f kcf_c(target_box.x + target_box.width / 2.0f, target_box.y + target_box.height / 2.0f);
            // 칼만이 예측했던 위치(predicted_pos)와 실제 KCF가 찾은 위치(kcf_c)의 거리(오차) 계산
            double error = cv::norm(predicted_pos - kcf_c); 
            std::cout << "   [Kalman Debug] Prediction Error: " << error << " pixels" << std::endl;
        }
        
        // [Step A] 전처리 수행 (컬러 -> 흑백 변환)
        cv::Mat processed_img;
        if (!preprocessor.process(current_frame, processed_img)) continue;

        // 칼만 필터(상태 추정) 핵심 테스트 로직
        cv::Point2f estimated_pos(0, 0);
        cv::Point2f estimated_vel(0,0);
        
        // [Step B] 시각화 준비
        cv::Mat display_img = current_frame.image.clone();
        
        if (isFound) {
            // KCF 추적 성공 시(Tracking)
            cv::Point2f kcf_center(target_box.x + target_box.width / 2.0f, target_box.y + target_box.height / 2.0f);
            conf_sum += conf;
            
            if (is_first_track || !state_estimator.isInitialized()) {
                // 첫 번째 추적 성공 시 또는 칼만 필터가 초기화되지 않은 경우: 상태 초기화
                state_estimator.initialize(kcf_center);
                is_first_track = false;
                estimated_pos = kcf_center;
                estimated_vel = cv::Point2f(0, 0);
            } else {
                // 이후 프레임에서는 칼만 필터 업데이트 수행
                estimated_pos = state_estimator.update(kcf_center);
                estimated_vel = state_estimator.getEstimatedVelocity();
            }

            // 추적 성공 시: 초록색 사각형과 신뢰도 표시
            cv::rectangle(display_img, target_box, cv::Scalar(0, 255, 0), 2);
            cv::putText(display_img, "STATE: TRACKING", cv::Point(15, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(display_img, "Conf: " + std::to_string(conf), 
                        cv::Point(15, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        } else {
            // 추적 실패 시 -> 칼만 필터 관성 에측치 사용
            estimated_pos = predicted_pos;
            estimated_vel = state_estimator.getEstimatedVelocity();

            // LOST 관련 시각화
            cv::putText(display_img, "STATE: LOST", cv::Point(15, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            
            if (state_estimator.isInitialized()) {
                // 칼만 필터가 초기화된 상태에서 LOST인 경우: 빨간색 원으로 관성 예측 위치 표시
                cv::circle(display_img, estimated_pos, 20, cv::Scalar(0, 0, 255), 2);
                cv::putText(display_img, "Predicted Pos", estimated_pos + cv::Point2f(10, -10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }
        }
        
        // --- 칼만 필터 추정 결과 시각화 (공통) ---
        if (state_estimator.isInitialized()) {
            // 최적 추정 위치 표시 (파란색 점)
            cv::circle(display_img, estimated_pos, 5, cv::Scalar(255, 0, 0), -1);
            
            // 속도 벡터 표시 (파란색 화살표)
            cv::Point2f velocity_vector_end = estimated_pos + estimated_vel * 0.2f; 
            cv::arrowedLine(display_img, estimated_pos, velocity_vector_end, cv::Scalar(255, 100, 0), 2);

            // 속도 텍스트 출력
            std::string vel_txt = "Vel: (" + std::to_string(static_cast<int>(estimated_vel.x)) + ", " + std::to_string(static_cast<int>(estimated_vel.y)) + ")";
            cv::putText(display_img, vel_txt, cv::Point(15, 90), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
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