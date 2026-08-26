#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <streambuf>
#include <filesystem>
#include <memory>
#include <cstdio>
#include <array>
#include <algorithm>

#include "input/VideoInput.hpp"
#include "common/Frame.hpp"
#include "preprocessing/Preprocessor.hpp"
#include "acquisition/AcquisitionManager.hpp"
#include "target_tracking/Tracker.hpp"
#include "state_estimation/StateEstimator.hpp"
#include "fsm/FsmModule.hpp"
#include "control/ControlCommand.hpp"
#include "hardware_output/ServoOutput.hpp"
#include "common/DebugConfig.hpp"

// ============================================================
// [로그 자동 저장] 콘솔에 찍히는 모든 출력을 화면에도 보여주는 동시에
// "<커밋해시>_<순번>_<영상이름>.log" 파일로도 자동 저장한다.
// 코드가 바뀔 때마다 로그 파일명에 그 시점의 git 커밋 해시가 남아서,
// 나중에 "이 로그가 어떤 코드로 만들어졌는지" 헷갈릴 일이 없다.
// ============================================================

// 현재 작업 트리의 git 커밋 해시(짧은 형태)를 조회. git이 없거나 실패하면 "nogit" 반환.
static std::string getGitCommitHash() {
    std::array<char, 64> buffer{};
    std::string result;

    FILE* pipe;
    #ifdef _WIN32
        pipe = _popen("git rev-parse --short HEAD 2>NUL", "r");
    #else
        pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
    #endif
        if (!pipe) return "nogit";

        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            result += buffer.data();
        }

    #ifdef _WIN32
        _pclose(pipe);
    #else
        pclose(pipe);
#endif

    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result.empty() ? "nogit" : result;
}

// std::cout/std::cerr에 찍히는 내용을 콘솔과 파일 두 곳에 동시에 기록하는 스트림버퍼
class TeeBuf : public std::streambuf {
public:
    TeeBuf(std::streambuf* b1, std::streambuf* b2) : m_buf1(b1), m_buf2(b2) {}

protected:
    // 문자열/버퍼 단위로 한 번에 넘어오는 쓰기는 문자 단위로 쪼개지 않고
    // 통째로 양쪽에 전달한다 (성능 핵심: operator<<의 대부분은 이 경로를 탄다).
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        std::streamsize r1 = m_buf1->sputn(s, n);
        std::streamsize r2 = m_buf2->sputn(s, n);
        return std::min(r1, r2);
    }

    // xsputn이 처리하지 못하고 넘어온 한 글자짜리 예외 케이스에 대한 폴백
    int overflow(int c) override {
        if (c == EOF) return !EOF;
        int r1 = m_buf1->sputc(static_cast<char>(c));
        int r2 = m_buf2->sputc(static_cast<char>(c));
        return (r1 == EOF || r2 == EOF) ? EOF : c;
    }

    int sync() override {
        int r1 = m_buf1->pubsync();
        int r2 = m_buf2->pubsync();
        return (r1 == 0 && r2 == 0) ? 0 : -1;
    }

private:
    std::streambuf* m_buf1;
    std::streambuf* m_buf2;
};

// RAII 로거: 생성되는 순간부터 소멸될 때까지의 모든 std::cout/std::cerr 출력을
// 자동으로 로그 파일에도 남긴다. main()의 다른 모든 지역 변수보다 먼저 생성해야
// (즉 맨 첫 줄에 선언해야) 프로그램 실행 전체를 빠짐없이 기록할 수 있다.
class FileLogger {
public:
    FileLogger(int argc, char** argv) {
        namespace fs = std::filesystem;

        std::string gitHash    = getGitCommitHash();
        std::string videoLabel = (argc > 1) ? fs::path(argv[1]).stem().string() : "camera";

        fs::path logsDir = "logs";
        std::error_code ec;
        fs::create_directories(logsDir, ec);

        // 같은 커밋 해시로 이미 저장된 로그 개수를 세어 순번을 자동으로 매긴다.
        std::string prefix = gitHash + "_";
        int nextNum = 1;
        if (!ec && fs::exists(logsDir)) {
            for (const auto& entry : fs::directory_iterator(logsDir)) {
                if (entry.is_regular_file() && entry.path().filename().string().rfind(prefix, 0) == 0) {
                    nextNum++;
                }
            }
        }

        char numBuf[8];
        std::snprintf(numBuf, sizeof(numBuf), "%03d", nextNum);

        m_logPath = (logsDir / (gitHash + "_" + numBuf + "_" + videoLabel + ".log")).string();
        m_file.open(m_logPath);

        if (m_file.is_open()) {
            m_coutTee = std::make_unique<TeeBuf>(std::cout.rdbuf(), m_file.rdbuf());
            m_cerrTee = std::make_unique<TeeBuf>(std::cerr.rdbuf(), m_file.rdbuf());
            m_oldCoutBuf = std::cout.rdbuf(m_coutTee.get());
            m_oldCerrBuf = std::cerr.rdbuf(m_cerrTee.get());
            std::cout << "[Logger] 로그 저장 경로: " << m_logPath << std::endl;
        } else {
            std::cerr << "[Logger] 로그 파일을 열 수 없습니다: " << m_logPath << std::endl;
        }
    }

    ~FileLogger() {
        // std::cout/std::cerr을 원래 버퍼로 되돌려놓아야, 이 객체(및 파일 스트림)가
        // 먼저 소멸된 뒤에도 다른 static 객체가 cout을 안전하게 쓸 수 있다.
        if (m_oldCoutBuf) std::cout.rdbuf(m_oldCoutBuf);
        if (m_oldCerrBuf) std::cerr.rdbuf(m_oldCerrBuf);
    }

private:
    std::ofstream            m_file;
    std::unique_ptr<TeeBuf>  m_coutTee;
    std::unique_ptr<TeeBuf>  m_cerrTee;
    std::streambuf*          m_oldCoutBuf = nullptr;
    std::streambuf*          m_oldCerrBuf = nullptr;
    std::string              m_logPath;
};

// 실행 인자 규격 (재현 가능한 회귀 테스트용):
//   인자 없음                          -> 라이브 카메라(0번) 사용, ROI는 마우스로 직접 선택
//   <영상경로>                         -> 해당 영상 파일 재생.
//                                        같은 폴더에 동일한 이름의 .roi 프리셋 파일이 있으면
//                                        (예: data/test1.mp4 -> data/test1.roi, 내용은 "x y w h")
//                                        그 좌표를 자동으로 고정 ROI로 사용하고, 없으면 기존처럼
//                                        마우스로 직접 선택한 뒤 그 좌표를 .roi 파일로 자동 저장해서
//                                        다음 실행부터는 동일 파일에 대해 마우스 선택 없이 재사용됨
//   <영상경로> <x> <y> <w> <h>         -> 영상 파일 재생 + ROI 좌표 고정 (커맨드라인 인자가 .roi 프리셋 파일보다 항상 우선, .roi 파일에 저장되지는 않음)

// data/testN.mp4 같은 영상 파일과 같은 폴더에, 같은 이름에 확장자만 .roi인 파일이 있으면
// "x y w h" 형식의 좌표를 읽어 고정 ROI로 사용한다. 우선순위는 커맨드라인 좌표 인자보다는
// 낮고 마우스 수동 선택보다는 높다. 파일이 없거나 형식이 잘못되면 false를 반환해서
// 기존 동작(마우스 선택)으로 자연스럽게 넘어가게 한다.
static bool tryLoadRoiPreset(const std::string& videoPath, cv::Rect& outBox) {
    std::filesystem::path presetPath = std::filesystem::path(videoPath).replace_extension(".roi");

    std::ifstream in(presetPath);
    if (!in.is_open()) {
        return false; // 프리셋 파일이 없으면 조용히 폴백 (에러 아님)
    }

    int x, y, w, h;
    if (!(in >> x >> y >> w >> h)) {
        std::cerr << "[Warning] ROI 프리셋 파일 형식이 올바르지 않습니다: " << presetPath.string()
                  << " (\"x y w h\" 정수 4개 필요). 마우스 선택으로 대체합니다." << std::endl;
        return false;
    }

    outBox = cv::Rect(x, y, w, h);
    std::cout << "[Init] ROI 프리셋 파일 로드: " << presetPath.string()
              << " -> " << outBox << std::endl;
    return true;
}

// 마우스로 ROI를 처음 선택했을 때, 다음 실행부터는 동일 파일에 대해 자동으로 재사용할 수
// 있도록 같은 이름의 .roi 파일로 저장한다. 이 함수는 프리셋 파일이 "없어서" selectROI로
// 넘어온 경우에만 호출되므로, 기존 프리셋 파일을 실수로 덮어쓸 위험은 없다.
static void saveRoiPreset(const std::string& videoPath, const cv::Rect& box) {
    std::filesystem::path presetPath = std::filesystem::path(videoPath).replace_extension(".roi");
    std::ofstream out(presetPath);
    if (!out.is_open()) {
        std::cerr << "[Warning] ROI 프리셋 파일을 저장하지 못했습니다: " << presetPath.string() << std::endl;
        return;
    }
    out << box.x << " " << box.y << " " << box.width << " " << box.height << std::endl;
    std::cout << "[Init] ROI 프리셋 파일 저장 (다음 실행부터 자동 재사용됨): " << presetPath.string()
              << " -> " << box << std::endl;
}

int main(int argc, char** argv) {
    // [검증용 스위치] 칼만 필터의 관성(예측 의존)이 "표적을 못 따라가는" 문제의 원인인지
    // 확인하기 위한 임시 진단용 플래그.
    // true  -> 칼만 보정값 대신 KCF 원시 측정 좌표를 그대로 서보/로그(estimated_pos)에 사용
    // false -> 원래 동작(칼만 보정값 사용)으로 복귀
    // 검증이 끝나면 false로 되돌리거나 이 스위치와 관련 분기를 제거할 것.
    const bool DEBUG_USE_RAW_KCF_POS = false;

    // 0. 이 시점부터의 모든 콘솔 출력을 로그 파일에도 자동 저장 (반드시 가장 먼저 생성)
    FileLogger file_logger(argc, argv);

    // [디버그] 파이프라인 중간 이미지 저장 폴더 준비.
    // AcquisitionManager/Tracker 등 다른 모듈에서도 이 폴더에 저장하므로 가장 먼저 생성해둔다.
    if (DebugConfig::kEnableImageDump) {
        std::error_code dbg_ec;
        std::filesystem::create_directories(DebugConfig::kDebugImageDir, dbg_ec);
    }

    // 1. 모듈 객체 생성
    VideoInput video_input;
    Preprocessor preprocessor;
    AcquisitionManager acq_manager;
    Tracker tracker;
    StateEstimator state_estimator;
    FsmModule fsm;
    ControlCommand control_command(0.02, 0.02, 0.002, 0.002);
    ServoOutput servo_output(0.0, 180.0, 30.0, 150.0);

    // 2. 영상 소스 열기: 인자로 영상 경로가 주어지면 파일 재생, 없으면 라이브 카메라
    bool useFile = (argc > 1);
    bool opened  = useFile ? video_input.openFile(argv[1]) : video_input.open(0);

    if (!opened) {
        std::cerr << "[Error] " << (useFile ? "영상 파일을 열 수 없습니다: " + std::string(argv[1])
                                             : "카메라를 열 수 없습니다.")
                  << std::endl;
        return -1;
    }

    std::cout << "=== EO Seeker 실행 (" << (useFile ? "파일 재생: " + std::string(argv[1]) : "라이브 카메라")
              << ") ===" << std::endl;

    Frame current_frame;
    cv::Rect target_box;

    // 3. 초기 ROI 설정을 위한 프레임 획득
    if (video_input.read(current_frame)) {
        // 좌표 4개(x y w h)가 함께 주어지면 마우스 선택 없이 고정 ROI 사용 (최우선)
        cv::Rect presetBox;
        if (argc > 5) {
            target_box = cv::Rect(std::stoi(argv[2]), std::stoi(argv[3]),
                                   std::stoi(argv[4]), std::stoi(argv[5]));
            std::cout << "[Init] 고정 ROI 사용 (커맨드라인 인자): " << target_box << std::endl;
        } else if (useFile && tryLoadRoiPreset(argv[1], presetBox)) {
            // 커맨드라인 좌표가 없으면, 같은 이름의 .roi 프리셋 파일을 시도
            target_box = presetBox;
        } else {
            target_box = cv::selectROI("Original & Tracking", current_frame.image, false);

            // 영상 파일 모드에서 마우스로 처음 선택한 경우, 다음부터 재사용할 수 있도록 자동 저장
            if (useFile && target_box.width > 0 && target_box.height > 0) {
                saveRoiPreset(argv[1], target_box);
            }
        }

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

            // [워밍업] KCF FFT 플랜 / ORB 검출기 내부 캐시를 세션 타이머 시작 전에 미리 예열
            // 동일한 정지 이미지로만 반복 실행하므로 실제 추적 결과에는 영향 없음 (1회성 초기화 비용을 세션 통계에서 분리하기 위함)
            {
                auto warmup_start = std::chrono::steady_clock::now();
                std::cout << "[Warm-up] 초기화 캐시 예열 중..." << std::endl;
                for (int i = 0; i < 6; ++i) {
                    tracker.update(initial_processed_img, acq_manager.getTargetTemplate(), acq_manager.getTargetDescriptors());
                }
                // 예열 중 누적된 프레임 카운터/신뢰도 상태를 깨끗하게 리셋 (실제 세션은 항상 동일한 초기 상태에서 시작)
                tracker.init(initial_processed_img, target_box,
                             acq_manager.getTargetTemplate(), acq_manager.getTargetDescriptors());
                double warmup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warmup_start).count();
                std::cout << "[Warm-up] 완료 (" << std::fixed << std::setprecision(0) << warmup_ms
                          << "ms, 세션 통계에서 제외됨)" << std::endl;
            }

            // 예열 후 새 프레임을 다시 읽어 실제 세션 시작 시점 기준으로 삼음 (세션 타이머가 워밍업 비용을 포함하지 않도록)
            if (!video_input.read(current_frame)) {
                std::cerr << "[Error] 예열 후 프레임을 읽어올 수 없습니다." << std::endl;
                return -1;
            }

            /*칼만 필터 초기화*/
            cv::Point2f kcf_center(target_box.x + target_box.width / 2.0f, target_box.y + target_box.height / 2.0f);
            state_estimator.initialize(kcf_center); // 칼만 필터 초기화

            fsm.setTargetBox(target_box);
            fsm.forceSetState(FSMState::TRACK); // 초기 상태 설정

            std::cout << "[Init] Target & Tracker Registered!" << std::endl;
            std::cout << "Initial Target Box info: \"" << target_box.x << "\", \"" << target_box.y << "\", \"" << target_box.width << "\", \"" << target_box.height << "\"" << std::endl;
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

        // [디버그] 전처리 결과(흑백 변환 후 이미지)를 10프레임마다 저장.
        // 노출/블러/색공간 문제로 인해 이후 단계(ORB, NCC)가 나빠지는 건 아닌지 확인용.
        if (DebugConfig::kEnableImageDump && current_frame.frame_count % 10 == 0) {
            cv::imwrite(DebugConfig::kDebugImageDir + "preprocessed_f" + std::to_string(current_frame.frame_count) + ".png",
                        processed_img);
        }

        /* FSM 제어부: 현재 상태에 따른 행동 제어 및 조건 처리 */
        switch (fsm.getCurrentState()) {
            
            case FSMState::TRACK: {
                // KCF 추적 수행
                // [재수정] 원본 고정 스냅샷(getOriginalTemplate/getOriginalDescriptors)과만
                // 비교하도록 했던 이전 변경을 되돌린다. 표적이 회전/드리프트하지 않아도
                // 빠른 이동으로 인한 모션 블러만으로 원본(정지 상태, 또렷함)과의 NCC 유사도가
                // 떨어지면서 KCF는 정상 추적 중(success=true)인데도 conf가 0으로 붕괴해
                // LOST로 잘못 전이되는 문제가 실측(로그+디버그 프레임)으로 확인됐다.
                // 드리프트 방지는 AcquisitionManager::isStillSimilarToOriginal이
                // (updateTargetModel 갱신을 허용할지 판단할 때) 원본과의 유사도를 여전히
                // 검증하므로, conf 계산까지 원본 고정 비교로 이중으로 엄격하게 걸 필요는 없다.
                // conf는 "최근 갱신된(=최근 흐려진 정도까지 반영된) 템플릿"과 비교해
                // 자세/블러 변화에는 관대하되, 갱신 자체는 원본 anchor 검증을 통과한 것만
                // 반영되므로 드리프트 방지 효과는 유지된다.
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

                    cv::Point2f kalman_corrected_pos = state_estimator.update(kcf_center);
                    // [검증용] 플래그가 켜져 있으면 칼만 보정값 대신 KCF 원시 좌표를 그대로 사용.
                    // state_estimator.update()는 그대로 호출해 내부 속도 추정 등은 계속 정상 갱신됨.
                    estimated_pos = DEBUG_USE_RAW_KCF_POS ? kcf_center : kalman_corrected_pos;
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

                // [디버그] REACQUIRE 검증 대상 후보 이미지 저장.
                // 실제로 표적을 담고 있는 영역인지, 아니면 배경/노이즈를 표적으로 착각한 건지 확인용.
                if (DebugConfig::kEnableImageDump) {
                    cv::imwrite(DebugConfig::kDebugImageDir + "reacquire_candidate_f" + std::to_string(current_frame.frame_count) + ".png",
                                candidateROI);
                }

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

        // [디버그] 칼만 보정 전 순수 예측 위치(predicted_pos)를 노란 점으로 별도 표시.
        // estimated_pos(파란 점, 보정값)와의 간격이 벌어질수록 예측이 실제 표적을 놓치고
        // 있다는 뜻이라, 탐색 반경/재획득 실패 원인을 눈으로 바로 판단하는 데 쓴다.
        cv::circle(display_img, predicted_pos, 4, cv::Scalar(0, 255, 255), -1);

            cv::putText(display_img, "F: " + std::to_string(current_frame.frame_count), cv::Point(current_frame.width - 100, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

        std::string cmd_text = "Pan Cmd: " + std::to_string(static_cast<int>(servo_cmd.pan_cmd)) + 
                                " | Tilt Cmd: " + std::to_string(static_cast<int>(servo_cmd.tilt_cmd));
        cv::putText(display_img, cmd_text, cv::Point(15, current_frame.height - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);                  
        
        cv::putText(display_img, "F: " + std::to_string(current_frame.frame_count), cv::Point(current_frame.width - 100, 30), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(255, 255, 0), 1);
        cv::imshow("Tracking Test", display_img);

        // [디버그] 최종 오버레이 프레임(bbox, FSM 상태, 추정/예측 위치)을 10프레임마다 저장.
        // 전체 파이프라인이 그 순간 실제로 뭘 보고 있었는지 최종 확인용.
        if (DebugConfig::kEnableImageDump && current_frame.frame_count % 10 == 0) {
            cv::imwrite(DebugConfig::kDebugImageDir + "final_overlay_f" + std::to_string(current_frame.frame_count) + ".png",
                        display_img);
        }

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
