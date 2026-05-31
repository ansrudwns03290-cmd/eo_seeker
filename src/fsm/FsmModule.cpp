#include "fsm/FsmModule.hpp"
#include <iostream>

FsmModule::FsmModule() 
    : m_currentState(FSMState::SEARCH)
    , m_targetBox(0, 0, 0, 0)
    , m_temporaryBox(0, 0, 0, 0)
    , m_lostStartTime(0)
    , m_lowConfidenceCounter(0)
    , m_isHighSpeedLoss(false) {
    std::cout << "[FSM] State Machine Initialized. Current: SEARCH" << std::endl;
}

FsmModule::~FsmModule() {}

FSMState FsmModule::getCurrentState() const {
    return m_currentState;
}

std::string FsmModule::getStateString() const {
    switch (m_currentState) {
        case FSMState::SEARCH:    return "SEARCH";
        case FSMState::TRACK:     return "TRACK";
        case FSMState::LOST:      return "LOST";
        case FSMState::REACQUIRE: return "REACQUIRE";
        default:                  return "UNKNOWN";
    }
}

cv::Rect FsmModule::getTargetBox() const {
    return m_targetBox;
}

void FsmModule::setTargetBox(const cv::Rect& box) {
    m_targetBox = box;
}

void FsmModule::forceSetState(FSMState newState) {
    logTransition(m_currentState, newState);
    m_currentState = newState;
    if (newState == FSMState::SEARCH) {
        m_lowConfidenceCounter = 0;
        m_targetBox = cv::Rect(0, 0, 0, 0);
    }
}

void FsmModule::logTransition(FSMState fromState, FSMState toState) {
    auto getStateName = [](FSMState state) {
        switch (state) {
            case FSMState::SEARCH:    return "SEARCH";
            case FSMState::TRACK:     return "TRACK";
            case FSMState::LOST:      return "LOST";
            case FSMState::REACQUIRE: return "REACQUIRE";
            default:                  return "UNKNOWN";
        }
    };
    std::cout << "[FSM Transition] " << getStateName(fromState) 
              << " -> " << getStateName(toState) << std::endl;
}

void FsmModule::update(const Frame& currentFrame, float trackerConfidence, bool kcfSuccess, const cv::Point2f& currentVelocity) {
    if (!currentFrame.is_valid) return;

    FSMState nextState = m_currentState;

    switch (m_currentState) {
        case FSMState::SEARCH: {
            // [SEARCH -> TRACK] 
            // 외부(main 루프)에서 acq_manager를 통해 유효한 표적이 최초 획득(Lock-On)되면 
            // main이 fsm.setTargetBox() 후 fsm.forceSetState(TRACK)로 전이시키므로
            // SEARCH 상태 내부에서는 본인 상태를 유지하며 자동/수동 입력을 대기합니다.
            break;
        }

        case FSMState::TRACK: {
            if (kcfSuccess && trackerConfidence >= 0.40) {
                // 정상 추적 상태 유지
                m_lowConfidenceCounter = 0; 
                nextState = FSMState::TRACK;
            } 
            else {
                // 순간 노이즈 방어용 카운터 누적
                m_lowConfidenceCounter++;
                
                // 연속 5프레임 이상 신뢰도가 떨어지면 마침내 상실 선언
                if (m_lowConfidenceCounter >= 5) {
                    m_lostStartTime = currentFrame.timestamp_ms; // 골든타임 타이머 시동
                    
                    // 칼만 필터에서 넘어온 속도 벡터의 크기 계산
                    // sqrt(vx^2 + vy^2)
                    double targetSpeed = std::sqrt(currentVelocity.x * currentVelocity.x +
                         currentVelocity.y * currentVelocity.y);
                    m_isHighSpeedLoss = (targetSpeed >= 15.0);
                    
                    nextState = FSMState::LOST;
                }
            }
            break;
        }

        case FSMState::LOST: {
            // 1. 골든타임 검증 (1.5초가 초과되면 미련 없이 초기 검색으로 복귀)
            int64_t elapsedTime = currentFrame.timestamp_ms - m_lostStartTime;
            if (elapsedTime > 1500) {
                std::cout << "[FSM Warning] 골든타임 1.5초 초과로 완전 상실 판정." << std::endl;
                m_lowConfidenceCounter = 0;
                m_targetBox = cv::Rect(0, 0, 0, 0);
                nextState = FSMState::SEARCH;
                break;
            }

            // 2. [사용자님 논리 반영]: LOST 상태에서 모터를 밀며 경량 고속 후보 탐색을 수행
            // 고속 상실 플래그에 따라 탐색 윈도우 크기를 가변 조정
            int searchWindowSize = m_isHighSpeedLoss ? 160 : 80;
            
            // 외부 main 파이프라인에서 acq_manager.findFastCandidate()를 호출한 결과
            // 후보군 물체가 감지되었음을 가정하는 가상 플래그 (통합 시 실제 함수로 대체)
            bool foundCandidate = true; 
            cv::Rect detectedCandidateBox(300, 200, 50, 50); // 예시 좌표

            // 1.5초 이내에 후보 윤곽선이 눈에 걸려들면 REACQUIRE(검증실) 상태로 전이
            if (foundCandidate) {
                m_temporaryBox = detectedCandidateBox; // 검증실로 넘겨줄 가짜 박스 보관
                nextState = FSMState::REACQUIRE;
            } else {
                // 후보가 없으면 타이머를 계속 째며 LOST 상태 유지 (칼만 예측 구동 명령 지속)
                nextState = FSMState::LOST;
            }
            break;
        }

        case FSMState::REACQUIRE: {
            // [사용자님 논리 반영]: 들어온 후보 박스 딱 1프레임 정밀 검증실 가동
            // 무거운 ORB/NCC 매칭 연산을 호출합니다.
            // (실전 구현 시 tracker.verifyTarget() 연산 점수를 대입하게 됩니다)
            float verificationScore = trackerConfidence; // 예시 합격 점수

            if (verificationScore >= 0.65) {
                std::cout << "[FSM Success] 표적 검증 통과! TRACK 상태로 복귀합니다." << std::endl;
                m_targetBox = m_temporaryBox; // 정답 좌표 확정
                m_lowConfidenceCounter = 0;   // 카운터 초기화
                nextState = FSMState::TRACK;  // 복귀 성공! (KCF 재부팅 플래그로 활용)
            } 
            else {
                // 모양이 전혀 다른 웅뚱한 노이즈(False Reacquire)였다면 즉시 SEARCH로 던져 시스템 안전 확보
                std::cout << "[FSM Fail] 가짜 노이프 판정 (" << verificationScore
                            << " ). LOST 상태 유지" << std::endl;

                nextState = FSMState::LOST;
            }
            break;
        }
    }

    // 상태 변화가 감지되었을 때만 로그 출력 및 상태 변경
    if (nextState != m_currentState) {
        logTransition(m_currentState, nextState);
        m_currentState = nextState;
    }
}