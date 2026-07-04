#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <string>

// KCF 추적기 단독 검증용 테스트
//
// AcquisitionManager(ORB/NCC 갱신), StateEstimator(칼만), FSM, confidence 계산 등
// 실제 파이프라인의 다른 로직을 전부 배제하고 cv::TrackerKCF 하나만 떼어내서,
// 같은 영상/같은 초기 박스로 표적을 얼마나 잘 따라가는지 확인하기 위한 프로그램이다.
//
// 실제 파이프라인(Preprocessor -> Tracker)에서 KCF가 실제로 보는 입력은
// "컬러 프레임을 흑백으로 바꾼 뒤 다시 3채널로 복원한 이미지"이므로(색상 정보 없음),
// 공정한 비교를 위해 여기서도 동일하게 변환해서 KCF에 넣는다.
//
// 사용법: test_kcf_standalone.exe <영상경로> [roi_x roi_y roi_w roi_h]
//   roi 인자를 생략하면 main.cpp에서 test2.mp4에 쓰던 기본 고정 ROI(382,190,175,189)를 사용한다.
//
// 관찰 포인트:
//   - "KCF Standalone Test" 창에서 초록 박스가 실제 표적을 계속 따라가는지 눈으로 확인.
//   - 콘솔에 찍히는 center 좌표가, 원래 파이프라인 로그에서 박스가 멈췄던 구간
//     (프레임 60~130 부근, (307~316, 268~273) 근방)에서도 똑같이 멈추는지 비교.
//   - update()가 false를 반환하는 지점이 있는지(있다면 KCF 스스로 실패를 인지한 것).
//     단, KCF는 배경에 눌러앉아도 내부적으로는 "성공"이라고 계속 우길 수 있으므로,
//     success 여부보다 bbox 궤적을 직접 보는 게 더 정확하다.

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "사용법: " << argv[0] << " <영상경로> [roi_x roi_y roi_w roi_h]" << std::endl;
        return -1;
    }

    std::string videoPath = argv[1];

    // main.cpp에서 test2.mp4에 쓰던 기본값과 동일
    //cv::Rect initBox(382, 190, 175, 189); // test2
    //cv::Rect initBox(416, 247, 180, 171); // test3
    cv::Rect initBox(405, 123, 173, 165); // test5
    
    if (argc >= 6) {
        initBox = cv::Rect(std::stoi(argv[2]), std::stoi(argv[3]),
                            std::stoi(argv[4]), std::stoi(argv[5]));
    }

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[Error] 영상을 열 수 없습니다: " << videoPath << std::endl;
        return -1;
    }

    // [중요] VideoInput::read()는 파일 재생 모드일 때 영상 원본 해상도와 무관하게
    // 항상 640x480으로 리사이즈한다(카메라 표준 사양과 동일 조건을 맞추기 위함).
    // main.cpp에서 쓰던 ROI 좌표(382,190,175,189)는 이 리사이즈된 640x480 프레임 기준이므로,
    // 여기서도 똑같이 리사이즈하지 않으면 같은 좌표를 줘도 실제로는 전혀 다른 위치/크기를
    // 가리키게 되어 ROI가 엉뚱하게 잡힌다.
    auto toStandardRes = [](cv::Mat& img) {
        if (img.cols != 640 || img.rows != 480) {
            cv::resize(img, img, cv::Size(640, 480));
        }
    };

    cv::Mat frame;
    if (!cap.read(frame)) {
        std::cerr << "[Error] 첫 프레임을 읽을 수 없습니다." << std::endl;
        return -1;
    }
    toStandardRes(frame);

    // 실제 파이프라인(Preprocessor::process + Tracker의 KCF 입력 변환)과 동일하게
    // 컬러 -> 그레이스케일 -> 3채널 복원 과정을 그대로 재현한다.
    auto toKcfInput = [](const cv::Mat& colorFrame) {
        cv::Mat gray, kcfInput;
        cv::cvtColor(colorFrame, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, kcfInput, cv::COLOR_GRAY2BGR);
        return kcfInput;
    };

    cv::Mat kcfInputFrame = toKcfInput(frame);

    cv::Rect safeBox = initBox & cv::Rect(0, 0, kcfInputFrame.cols, kcfInputFrame.rows);
    if (safeBox.width <= 0 || safeBox.height <= 0) {
        std::cerr << "[Error] 초기 ROI가 영상 범위를 벗어났습니다: " << initBox << std::endl;
        return -1;
    }

    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    tracker->init(kcfInputFrame, safeBox);

    std::cout << "=== KCF 단독 추적 테스트 시작 ===" << std::endl;
    std::cout << "영상: " << videoPath << std::endl;
    std::cout << "초기 박스: " << safeBox << std::endl;

    cv::Rect trackedBox = safeBox;
    int frameCount = 0;
    int firstFailFrame = -1;

    while (cap.read(frame)) {
        frameCount++;
        toStandardRes(frame);
        kcfInputFrame = toKcfInput(frame);

        bool ok = tracker->update(kcfInputFrame, trackedBox);

        cv::Point2f center(trackedBox.x + trackedBox.width / 2.0f,
                            trackedBox.y + trackedBox.height / 2.0f);

        std::cout << "Frame: " << frameCount
                  << " | success: " << (ok ? "true" : "false")
                  << " | bbox: " << trackedBox
                  << " | center: (" << static_cast<int>(center.x) << ", " << static_cast<int>(center.y) << ")"
                  << std::endl;

        if (!ok && firstFailFrame == -1) {
            firstFailFrame = frameCount;
            std::cout << "[KCF] update()가 false를 반환한 최초 지점. Frame: " << frameCount << std::endl;
        }

        // 시각화: 원본 컬러 프레임 위에 KCF 원시 박스만 표시 (다른 로직 없음)
        cv::Mat display = frame.clone();
        cv::rectangle(display, trackedBox, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "Frame: " + std::to_string(frameCount), cv::Point(15, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        cv::imshow("KCF Standalone Test", display);

        int key = cv::waitKey(1);
        if (key == 27) break; // ESC로 중단
    }

    std::cout << "=== 테스트 종료 (총 " << frameCount << "프레임) ===" << std::endl;
    if (firstFailFrame != -1) {
        std::cout << "[결과] update()가 false를 반환한 최초 지점: Frame " << firstFailFrame << std::endl;
    } else {
        std::cout << "[결과] 전체 구간에서 update()는 계속 true를 반환함. "
                  << "(success 여부와 무관하게, 위 로그의 center 좌표 궤적과 화면 박스를 직접 확인할 것)"
                  << std::endl;
    }

    return 0;
}
