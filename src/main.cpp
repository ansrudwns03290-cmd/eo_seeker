#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>

#include "input/VideoInput.hpp"
#include "common/Frame.hpp"
#include "preprocessing/Preprocessor.hpp"
#include "acquisition/AcquisitionManager.hpp"
#include "target_tracking/Tracker.hpp"
#include "state_estimation/StateEstimator.hpp"
#include "fsm/FsmModule.hpp"
#include "control/ControlCommand.hpp"
#include "hardware_output/ServoOutput.hpp"

int main() {
    // 1. 모듈 객체 생성
    VideoInput video_input;
    Preprocessor preprocessor;
    AcquisitionManager acq_manager;
    Tracker tracker;
    StateEstimator state_estimator;
    FsmModule fsm;
    ControlCommand control_command(0.02, 0.02, 0.002, 0.002);
    ServoOutput servo_output(0.0, 180.0, 30.0, 150.0);
    
    // 2. 카메라 열기
    if (!video_input.open(0)) {
        std::cerr << "[Error] 카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    std::cout << "=== EO Seeker 실행 ===" << std::endl;

    Frame current_frame;
    cv::Rect target_box;

    // 3. 초기 ROI 설정을 위한 프레임 획득
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

            if (!tracker.init(initial_processed_img, target_box,
                              acq_manager.getTargetTemplate(), acq_manager.getTargetDescriptors())) {
                std::cerr << "[Error] Tracker 초기화 실패!" << std::endl;
                return -1;
            }

            /*칼만 필터 초기화*/
            cv::Point2f kcf_center(target_box.x + target_box.width / 2.0f, target_box.y + target_box.height / 2.0f);
            state_estimator.initialize(kcf_center); // 칼만 필터 초기화    
            
            fsm.setTargetBox(target_box);
            fsm.forceSetState(FSMState::TRACK); // 초기 상태 설정

            std::cout << "[Init] Target & Tracker Registered!" << std::endl;
            std::cout << "Initial Target Box size: " << target_box.width << " x " << target_box.height << std::endl;
        }
    } else {
        std::cout << "[Error] 초기 프레임을 읽어올 수 없습니다." << std::endl;
        return -1;
    }

    // 2. 루프 시작
    double last_timestamp_ms = current_frame.timestamp_ms;
    double session_start_ms  = current_frame.timestamp_ms; // 상대 시각 기준점

    // 세션 통계 변수
    int   stat_track_frames      = 0;
    int   stat_lost_count        = 0;
    int   stat_reacquire_success = 0;
    int   stat_reacquire_fail    = 0;
    FSMState prev_fsm_state      = FSMState::TRACK;

    // FPS 계산용
    double  current_fps  = 0.0;
    int64_t fps_ref_ts   = static_cast<int64_t>(session_start_ms);

    // [진단용] 카메라 대기 시간 vs 처리(추적+FSM+표시) 시간 분리 계측
    double accum_read_ms    = 0.0;
    double accum_process_ms = 0.0;
    auto   t_after_read     = std::chrono::steady_clock::now();

    cv::Point2f estimated_pos(0, 0);
    cv::Point2f estimated_vel(0,0);

    while (true) {
        auto t_read_start = std::chrono::steady_clock::now();
        if (!video_input.read(current_frame)) break;
        t_after_read = std::chrono::steady_clock::now();
        accum_read_ms += std::chrono::duration<double, std::milli>(t_after_read - t_read_start).count();

        // --- 칼만 필터 예측 수행 ---
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
                // KCF 추적 수행
                Tracker::TrackingResult res = tracker.update(processed_img,
                acq_manager.getTargetTemplate(), acq_manager.getTargetDescriptors());
                isFound = res.success;

                conf = tracker.getConfidence();

                if (isFound && conf >= 0.40) {
                    if (res.needTemplateUpdate) {
                        acq_manager.updateTargetModel(res.newTemplate, res.newDescriptors);
                    }

                    // 추적 성공 시 -> 칼만 필터 보정 및 타겟 박스 확정
                    cv::Point2f kcf_center(res.bbox.x + res.bbox.width / 2.0f, 
                                           res.bbox.y + res.bbox.height / 2.0f);

                    estimated_pos = state_estimator.update(kcf_center);
                    estimated_vel = state_estimator.getEstimatedVelocity();

                    fsm.setTargetBox(res.bbox);
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
                
                std::cout << "DEBUG 1: LOST 상태에서 후보 탐색 시작. 예측 위치 중심: (" << static_cast<int>(estimated_pos.x) << ", " << static_cast<int>(estimated_pos.y) << ")" << std::endl;
                bool foundCandidate = acq_manager.detectCandidateInPredictArea(
                    processed_img,            // 1. 전처리 모듈이 만든 흑백 영상
                    safe_coast_box,           // 2. 칼만이 예측한 안전 영역 사각형
                    fsm.getSearchWindowSize(), // 3. FSM 내부 알고리즘이 결정한 동적 윈도우 크기 (80 또는 160)
                    candidateBox              // 4. [출력] 새로 찾아낸 후보 좌표를 받아올 변수
                );
                
                std::cout << "DEBUG 2: 후보 탐색 결과 = " << (foundCandidate ? "발견" : "미발견") << std::endl;
                
                if (foundCandidate) {
                    // 후보 발견된 경우 다음 프레임에 REACQUIRE 상태에서 검증하기 위해 플래그 설정
                    isFound = true; // 현재 프레임에서는 아직 KCF 구동하지 않음
                    conf = 0.0f; // 임시 합격 커트라인 점수를 주어 FSM의 update 스위치 동작
                    
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
            
                // 디버깅: 후보 이미지 저장
                
                // cv::imwrite("C:/eo_seeker/debug_images/candidate_roi_" + std::to_string(current_frame.frame_count) + ".png", candidateROI);
                std::cout << "후보 이미지 저장" << std::endl;
                
                float v_score = tracker.verifyCandidate(candidateROI, acq_manager.getTargetDescriptors(),
                                                        acq_manager.getTargetTemplate()); // 후보 검증 수행 (KCF 기반)
                std::cout << "DEBUG: 후보 검증 점수 = " << v_score << std::endl;

                if (v_score >= 0.60f) {
                    stat_reacquire_success++;
                    // 검증 통과 시 추적기 새 위치로 재부팅
                    tracker.init(current_frame.image, tempBox, acq_manager.getTargetTemplate(), acq_manager.getTargetDescriptors());
                    cv::Point2f re_center(tempBox.x + tempBox.width / 2.0f, tempBox.y + tempBox.height / 2.0f);
                    
                    state_estimator.update(re_center); // 칼만 필터도 새 위치로 보정

                    estimated_vel = state_estimator.getEstimatedVelocity(); // 보정된 속도 벡터 업데이트
                    // FSM 강제 복귀 처리용 플래그
                    isFound = true;
                    conf = v_score;
                }else {
                    stat_reacquire_fail++;
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

        // 세션 통계 수집
        if (fsm.getCurrentState() == FSMState::TRACK) stat_track_frames++;
        if (prev_fsm_state == FSMState::TRACK && fsm.getCurrentState() == FSMState::LOST) stat_lost_count++;
        prev_fsm_state = fsm.getCurrentState();

        // 제어 명령 생성 모듈 구동
        // 칼만 필터가 산출한 정밀 최적 중심 위치와 현재 시스템 상태 문자열 주입
        ServoCommand servo_cmd = control_command.calculateCommand(estimated_pos.x, estimated_pos.y, fsm.getStateString());
        servo_output.sendCommand(servo_cmd);

        // FPS 계산 (10프레임마다 갱신)
        if (current_frame.frame_count % 10 == 0) {
            double elapsed_10f = (current_frame.timestamp_ms - fps_ref_ts) / 1000.0;
            current_fps = (elapsed_10f > 0.0) ? (10.0 / elapsed_10f) : 0.0;
            fps_ref_ts  = static_cast<int64_t>(current_frame.timestamp_ms);

            // [진단용] 최근 10프레임 평균 "카메라 대기 시간" vs "처리 시간" 출력
            std::cout << "[Timing] Camera Wait Avg: " << std::fixed << std::setprecision(2) << (accum_read_ms / 10.0)
                      << "ms | Processing Avg: " << (accum_process_ms / 10.0) << "ms (last 10 frames)" << std::endl;
            accum_read_ms    = 0.0;
            accum_process_ms = 0.0;
        }
        double rel_ts = current_frame.timestamp_ms - session_start_ms;

        if (fsm.getCurrentState() == FSMState::TRACK) {
            if (current_frame.frame_count % 10 == 0) {
                std::cout << "Frame: " << current_frame.frame_count
                  << " | [Tracking] Conf: " << std::fixed << std::setprecision(2) << conf
                  << " | FPS: " << std::fixed << std::setprecision(1) << current_fps
                  << " | TS: " << static_cast<int>(rel_ts) << "ms"
                  << " | Target Pos: (" << static_cast<int>(estimated_pos.x) << ", " << static_cast<int>(estimated_pos.y) << ")"
                  << " | [Servo Cmd] Pan: " << std::fixed << std::setprecision(2) << servo_cmd.pan_cmd
                  << " | Tilt: " << servo_cmd.tilt_cmd
                  << std::endl;
            }
        } else {
            std::cout << "Frame: " << current_frame.frame_count
                  << " | [State: " << fsm.getStateString() << "] Conf: " << std::fixed << std::setprecision(2) << conf
                  << " | TS: " << static_cast<int>(rel_ts) << "ms"
                  << " | Target Pos: (" << static_cast<int>(estimated_pos.x) << ", " << static_cast<int>(estimated_pos.y) << ")"
                  << " | [Servo Cmd] Pan: " << std::fixed << std::setprecision(2) << servo_cmd.pan_cmd
                  << " | Tilt: " << servo_cmd.tilt_cmd
                  << std::endl;
        }
        // }
        // // 메타데이터 확인 로그 (Resolution, Timestamp)
        // std::cout << "Frame: " << current_frame.width << "x" << current_frame.height
        //           << " | Count: " << current_frame.frame_count
        //           << " | State: " << fsm.getStateString()
        //           << " | Conf: " << std::fixed << std::setprecision(2) << conf
        //           << " | TS: " << current_frame.timestamp_ms << "ms" 
        //           << " | Vel: (" << static_cast<int>(estimated_vel.x) << ", " << static_cast<int>(estimated_vel.y) << ")"
        //           << std::endl;
        
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

        std::string cmd_text = "Pan Cmd: " + std::to_string(static_cast<int>(servo_cmd.pan_cmd)) + 
                                " | Tilt Cmd: " + std::to_string(static_cast<int>(servo_cmd.tilt_cmd));
        cv::putText(display_img, cmd_text, cv::Point(15, current_frame.height - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);                  
        
        cv::putText(display_img, "F: " + std::to_string(current_frame.frame_count), cv::Point(current_frame.width - 100, 30), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(255, 255, 0), 1);
        cv::imshow("Tracking Test", display_img);

        if (cv::waitKey(1) == 27) break; // ESC 누르면 수동 안전 종료

        // [진단용] 이번 프레임의 "카메라 대기"를 제외한 나머지(전처리~표시~waitKey) 소요 시간 누적
        accum_process_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_after_read).count();
    }

CORE_LOOP_EXIT:
    {
        double elapsed_sec = (last_timestamp_ms - session_start_ms) / 1000.0;
        int    total_frames = current_frame.frame_count;
        double avg_fps      = (elapsed_sec > 0.0) ? (total_frames / elapsed_sec) : 0.0;
        double track_rate   = (total_frames > 0)  ? (100.0 * stat_track_frames / total_frames) : 0.0;

        std::cout << "\n===== 세션 요약 =====" << std::endl;
        std::cout << "총 프레임    : " << total_frames << "프레임" << std::endl;
        std::cout << "총 경과 시간 : " << std::fixed << std::setprecision(1) << elapsed_sec << "s" << std::endl;
        std::cout << "평균 FPS     : " << std::fixed << std::setprecision(1) << avg_fps << std::endl;
        std::cout << "추적 성공률  : " << std::fixed << std::setprecision(1) << track_rate << "%" << std::endl;
        std::cout << "LOST 발생    : " << stat_lost_count << "회" << std::endl;
        std::cout << "재획득 성공  : " << stat_reacquire_success << "회" << std::endl;
        std::cout << "재획득 실패  : " << stat_reacquire_fail    << "회" << std::endl;
        std::cout << "=====================" << std::endl;
    }
    std::cout << "[System] 루프 종료" << std::endl;
    video_input.release();
    cv::destroyAllWindows();
    return 0;
}