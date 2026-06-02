#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "input/VideoInput.hpp"
#include "common/Frame.hpp"
#include "preprocessing/Preprocessor.hpp"
#include "acquisition/AcquisitionManager.hpp"
#include "target_tracking/Tracker.hpp"
#include "state_estimation/StateEstimator.hpp"
#include "fsm/FsmModule.hpp"


int main() {
    // 1. 모듈 객체 생성
    VideoInput video_input;
    Preprocessor preprocessor;
    AcquisitionManager acq_manager;
    Tracker tracker;
    StateEstimator state_estimator;
    FsmModule fsm;

    // 2. 카메라 열기
    if (!video_input.open(0)) {
        std::cerr << "[Error] 카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    std::cout << "=== EO Seeker 실행 ===" << std::endl;

    Frame current_frame;
    cv::Rect target_box;

    // 1. 초기 ROI 설정을 위한 프레임 획득
    if (video_input.read(current_frame)) {
        target_box = cv::selectROI("Original & Tracking", current_frame.image, false);
        
        if (target_box.width > 0 && target_box.height > 0) {
            /*tracker 초기화 및 초기 표적 등록*/
            // 이미지 경계 안전 처리
            cv::Rect img_rect(0, 0, current_frame.width, current_frame.height);
            target_box = target_box & img_rect;

            // 최초 프레임에 대해서도 전처리 수행
            cv::Mat initial_processed_img;
            if (!preprocessor.process(current_frame, initial_processed_img)) {
                std::cerr << "[Error] 최초 프레임 전처리 실패!" << std::endl;
                return -1;
            }

            // 표적 모델 등록 (이미지 조각 전달)
            cv::Mat roiImg = initial_processed_img(target_box);
            acq_manager.setTargetModel(roiImg);

            tracker.setTargetDescriptors(acq_manager.getTargetDescriptors()); 
            tracker.setTargetTemplate(roiImg);

            if (!tracker.init(initial_processed_img, target_box)) {
                std::cerr << "[Error] Tracker 초기화 실패!" << std::endl;
                return -1;
            }

            /*칼만 필터 초기화*/
            cv::Point2f kcf_center(target_box.x + target_box.width / 2.0f, target_box.y + target_box.height / 2.0f);
            state_estimator.initialize(kcf_center); // 칼만 필터 초기화    
            
            fsm.setTargetBox(target_box);
            fsm.forceSetState(FSMState::TRACK); // 초기 상태 설정

            std::cout << "[Init] Target & Tracker Registered!" << std::endl;
        }
    } else {
        std::cout << "[Error] 초기 프레임을 읽어올 수 없습니다." << std::endl;
        return -1;
    }

    // 2. 루프 시작
    double last_timestamp_ms = current_frame.timestamp_ms;
    cv::Point2f estimated_pos(0, 0);
    cv::Point2f estimated_vel(0,0);

    while (true) {
        if (!video_input.read(current_frame)) break;
        
        /*칼만 필터 예측 수행*/
        // 프레임 간 시간 간격(dt) 계산
        // 프레임 메타데이터의 구조적 timstap_ms를 초(seconds) 단위로 변환
        if (last_timestamp_ms == 0.0) {
            last_timestamp_ms = current_frame.timestamp_ms;
        }
        double dt = (current_frame.timestamp_ms - last_timestamp_ms) / 1000.0; // ms -> s
        last_timestamp_ms = current_frame.timestamp_ms;

        if (dt <= 0.0) dt = 0.033; // 프레임 간격이 비정상적으로 짧거나 없는 경우 기본값(30fps) 사용

        // 칼만 필터 예측 단계 선행 (매 프레임 무조건 먼저 수행)
        cv::Point2f predicted_pos = state_estimator.predict(dt);

        bool isFound = false;
        float conf = 0.0f;

        // --- 전처리 수행: 컬러 프레임 받아서 흑백 메트릭스 생성
        cv::Mat processed_img;
        if (!preprocessor.process(current_frame, processed_img)) {
            std::cout <<"[Warning] 프레임 전처리에 실패했습니다. 다음 프레임으로 넘어갑니다." << std::endl;
            
            if (current_frame.image.channels() == 3){
                cv::cvtColor(current_frame.image, processed_img, cv::COLOR_BGR2GRAY);
            } else {
                processed_img = current_frame.image.clone();
            }
        }

        /* FSM 제어부: 현재 상태에 따른 행동 제어 및 조건 처리 */
        switch (fsm.getCurrentState()) {
            
            case FSMState::TRACK: {
                // KCF를 구동하기 전에, target_box의 중심점을 칼만이 예측한 물리적 위치로 강제 이동시킵니다.
                target_box.x = predicted_pos.x - target_box.width / 2.0f;
                target_box.y = predicted_pos.y - target_box.height / 2.0f;
                
                // 이미지 경계 안전 처리 (화면 밖으로 나가는 것 방지)
                cv::Rect img_rect(0, 0, current_frame.width, current_frame.height);
                target_box = target_box & img_rect;

                // KCF 추적 수행
                isFound = tracker.update(processed_img, target_box);
                conf = tracker.getConfidence();

                if (isFound && conf >= 0.40) {
                    // 추적 성공 시 -> 칼만 필터 보정 및 타겟 박스 확정
                    cv::Point2f kcf_center(target_box.x + target_box.width / 2.0f, 
                                           target_box.y + target_box.height / 2.0f);

                    estimated_pos = state_estimator.update(kcf_center);
                    estimated_vel = state_estimator.getEstimatedVelocity();

                    fsm.setTargetBox(target_box);
                } else{
                    // 추적 실패 시 임시 관성 유지 처리
                    estimated_pos = predicted_pos;
                    estimated_vel = state_estimator.getEstimatedVelocity(); // 예측 단계에서 이미 속도는 계산되어 있음
                }
                break;
            }

            case FSMState::LOST: {
                // LOST 상태에서 KCF 구동하지 않고, 칼만 필터의 관성 위치로 박스 외형만 진행 (Coast Tracking)
                estimated_pos = predicted_pos;
                estimated_vel = state_estimator.getEstimatedVelocity();

                cv::Rect coast_box(estimated_pos.x - target_box.width / 2.0f,
                                   estimated_pos.y - target_box.height / 2.0f,
                                   target_box.width, target_box.height);
                
                cv::Rect safe_coast_box = coast_box & cv::Rect(0, 0, current_frame.width, current_frame.height);
                fsm.setTargetBox(coast_box & cv::Rect(0, 0, current_frame.width, current_frame.height));

                // 이진화 윤곽선 매칭을 이용한 고속 후보 탐색
                cv::Rect candidateBox;
                
                bool foundCandidate = acq_manager.detectCandidateInPredictArea(
                    processed_img,            // 1. 전처리 모듈이 만든 흑백 영상
                    safe_coast_box,           // 2. 칼만이 예측한 안전 영역 사각형
                    fsm.getSearchWindowSize(), // 3. FSM 내부 알고리즘이 결정한 동적 윈도우 크기 (80 또는 160)
                    candidateBox              // 4. [출력] 새로 찾아낸 후보 좌표를 받아올 변수
                );
                
                if (foundCandidate) {
                    // 후보 발견된 경우 다음 프레임에 REACQUIRE 상태에서 검증하기 위해 플래그 설정
                    isFound = false; // 현재 프레임에서는 아직 KCF 구동하지 않음
                    conf = 0.6f; // 임시 합격 커트라인 점수를 주어 FSM의 update 스위치 동작
                    
                    fsm.setTargetBox(candidateBox);
                } else {
                    isFound = false;
                    conf = 0.0f;
                }
                break;
            }

            case FSMState::REACQUIRE: {
                estimated_pos = predicted_pos;
                estimated_vel = state_estimator.getEstimatedVelocity();

                cv::Rect tempBox = fsm.getTargetBox(); //LOST에서 전달된 최종 관성 박스
                tempBox = tempBox & cv::Rect(0, 0, current_frame.width, current_frame.height); // 이미지 경계 안전 처리

                cv::Mat candidateROI = processed_img(tempBox); // 후보 영역 이미지 조각 추출
                
                float v_score = tracker.verifyCandidate(candidateROI); // 후보 검증 수행 (KCF 기반)
                std::cout << "DEBUG: 후보 검증 점수 = " << v_score << std::endl;

                if (v_score >= 0.65f) {
                    // 검증 통과 시 추적기 새 위치로 재부팅
                    tracker.init(current_frame.image, tempBox);
                    cv::Point2f re_center(tempBox.x + tempBox.width / 2.0f, tempBox.y + tempBox.height / 2.0f);
                    
                    state_estimator.update(re_center); // 칼만 필터도 새 위치로 보정

                    estimated_vel = state_estimator.getEstimatedVelocity(); // 보정된 속도 벡터 업데이트
                    // FSM 강제 복귀 처리용 플래그
                    isFound = true;
                    conf = v_score;
                }else {
                    // 노이즈인 경우 FSM이 다음 프레임에서 SEARCH로 전이되도록 유도
                    isFound = false;
                    conf = 0.0f;
                }
                break;
            }

            case FSMState::SEARCH: {
                // LOST 상태에서 1.5초 골든타임 오버 시 FSM 내부 update문에서 SEARCH로 강제 진입함
                // 메인 중앙 제어 규칙에 의거, 리셋 처리를 위해 break하여 안전 해제 구역으로 탈출
                std::cout << "[System Central Control] 표적 완전 상실로 제어를 종료합니다." << std::endl;
                goto CORE_LOOP_EXIT; // 이중 루프 또는 제어권 완전 탈출을 위한 정석적인 goto 핸들링
            }
        }

        fsm.update(current_frame, conf, isFound, estimated_vel); // FSM 상태 업데이트 (매 프레임마다 현재 프레임의 추적 성공 여부와 신뢰도 점수를 전달)

        // 메타데이터 확인 로그 (Resolution, Timestamp)
        std::cout << "Frame: " << current_frame.width << "x" << current_frame.height
                  << " | Count: " << current_frame.frame_count
                  << " | State: " << fsm.getStateString()
                  << " | Conf: " << std::fixed << std::setprecision(2) << conf
                  << " | TS: " << current_frame.timestamp_ms << "ms" 
                  << " | Vel: (" << static_cast<int>(estimated_vel.x) << ", " << static_cast<int>(estimated_vel.y) << ")"
                  << std::endl;
        
        // [Step B] 시각화 준비
        cv::Mat display_img = current_frame.image.clone();
        
        // --- 실시간 모니터링 그래픽 시각화 ---
        cv::Rect final_draw_box = fsm.getTargetBox();
        if (fsm.getCurrentState() == FSMState::TRACK) {
            cv::rectangle(display_img, final_draw_box, cv::Scalar(0, 255, 0), 2);
            cv::putText(display_img, "STATE: TRACKING", cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        } else if (fsm.getCurrentState() == FSMState::LOST || fsm.getCurrentState() == FSMState::REACQUIRE) {
            cv::circle(display_img, estimated_pos, 20, cv::Scalar(0, 0, 255), 2);
            cv::putText(display_img, "STATE: " +fsm.getStateString(), cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1); 
        }
        
        // --- 최적 추정 위치 및 속도 벡터 화살표 추력 ---
        cv::circle(display_img, estimated_pos, 5, cv::Scalar(255, 0, 0), -1);
        cv::Point2f velocity_vector_end = estimated_pos + estimated_vel * 0.2f; 
        cv::arrowedLine(display_img, estimated_pos, velocity_vector_end, cv::Scalar(255, 100, 0), 2);

        cv::putText(display_img, "F: " + std::to_string(current_frame.frame_count), cv::Point(current_frame.width - 100, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        cv::imshow("Tracking Test", display_img);

        if (cv::waitKey(1) == 27) break; // ESC 누르면 수동 안전 종료
    }

CORE_LOOP_EXIT:
    std::cout << "[System] 루프 종료" << std::endl;
    video_input.release();
    cv::destroyAllWindows();
    return 0;
}