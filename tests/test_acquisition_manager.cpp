#include <gtest/gtest.h>
#include "acquisition/AcquisitionManager.hpp"

namespace {

// setTargetModel()은 ROI 전체가 아니라 Otsu 이진화 + findContours로 찾은
// "가장 큰 물체" bbox만 잘라서 그 위에 ORB를 돌린다.
//
// 체커보드처럼 규칙적인 격자 무늬는 모든 교차점이 새들 포인트(대각선으로
// 밝고 어둡게 교차하는 X자 패턴)라서 FAST/ORB 코너 검출기가 코너로 인식하지
// 못한다 (체커보드 코너 전용 검출기인 cv::findChessboardCorners가 따로
// 존재하는 이유이기도 하다). 그래서 배경(50)보다 확실히 밝은 값 구간
// (140~220) 안에서 픽셀 단위로 흔들리는 노이즈 텍스처를 사용해, 물체 내부에
// ORB가 실제로 코너로 인식할 만한 밝기 변화를 다수 만든다. objectSize를
// 넉넉히(150px) 잡아 ORB edgeThreshold(31px) 가장자리 제외 영역을 피해서도
// 충분한 코너가 남도록 했다. Python(cv2)으로 사전 시뮬레이션해 106개의
// keypoints가 검출됨을 확인했다.
cv::Mat makeTexturedObjectRoi(int canvasSize = 200, int objectSize = 150) {
    cv::Mat img(canvasSize, canvasSize, CV_8UC3, cv::Scalar(50, 50, 50));

    cv::Mat noise(objectSize, objectSize, CV_8UC1);
    cv::RNG rng(12345);  // 테스트 재현성을 위한 고정 시드
    rng.fill(noise, cv::RNG::UNIFORM, 140, 221);

    cv::Mat noise3;
    cv::cvtColor(noise, noise3, cv::COLOR_GRAY2BGR);

    int offset = (canvasSize - objectSize) / 2;
    noise3.copyTo(img(cv::Rect(offset, offset, objectSize, objectSize)));

    return img;
}

}  // namespace

TEST(AcquisitionManagerTest, SetTargetModelExtractsDescriptorsFromTexturedImage) {
    AcquisitionManager acq;
    cv::Mat roi = makeTexturedObjectRoi();

    acq.setTargetModel(roi);

    EXPECT_FALSE(acq.getTargetTemplate().empty());
    EXPECT_GT(acq.getTargetDescriptors().rows, 0);
}

TEST(AcquisitionManagerTest, DetectCandidateFailsWithoutTargetModel) {
    AcquisitionManager acq;  // setTargetModel을 아직 호출하지 않아 m_targetTemplate이 비어있음

    cv::Mat gray(240, 320, CV_8UC1, cv::Scalar(50));
    cv::Rect predictArea(100, 80, 60, 60);
    cv::Rect outBox;

    EXPECT_FALSE(acq.detectCandidateInPredictArea(gray, predictArea, 80, outBox));
}

TEST(AcquisitionManagerTest, DetectCandidateFailsOnInvalidPredictRect) {
    AcquisitionManager acq;
    cv::Mat roi = makeTexturedObjectRoi();
    acq.setTargetModel(roi);

    cv::Mat gray(240, 320, CV_8UC1, cv::Scalar(50));
    cv::Rect invalidRect(0, 0, 0, 0);  // width/height 0
    cv::Rect outBox;

    EXPECT_FALSE(acq.detectCandidateInPredictArea(gray, invalidRect, 80, outBox));
}
