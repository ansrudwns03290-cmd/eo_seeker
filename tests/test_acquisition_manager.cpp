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

// 표적이 프레임 경계로 이탈해 탐색 창이 템플릿보다 작아지는 경우
// (재현: move_h_mid_01 프레임 109). 크래시 없이 미발견(false)이어야 한다.
TEST(AcquisitionManagerTest, DetectCandidateReturnsFalseWhenSearchWindowSmallerThanTemplate) {
    AcquisitionManager acq;
    acq.setTargetModel(makeTexturedObjectRoi());
    const cv::Mat& tmpl = acq.getTargetTemplate();
    ASSERT_FALSE(tmpl.empty());

    cv::Mat gray(240, 320, CV_8UC1, cv::Scalar(50));
    // 예측 위치를 우하단 모서리 밖으로 → 경계 교집합 후 탐색 창이 템플릿보다 작아짐
    cv::Rect predictArea(130, 220, 60, 60);
    cv::Rect outBox;

    bool found = true;
    EXPECT_NO_THROW(found = acq.detectCandidateInPredictArea(gray, predictArea, 200, outBox));
    EXPECT_FALSE(found);
}

// ---------------------------------------------------------------------------
// [P0] 템플릿 갱신 승인 차단: 디스크립터가 kMinUpdateDescriptors(7) 미만이면
// 갱신 전체(템플릿 + 디스크립터)를 거부하고 이전 모델을 유지해야 한다.
// 근거: verifyTarget / verifyCandidate는 디스크립터 7개 미만이면 ORB를 쓰지 않고
//       NCC 단독으로 판정한다 → 7개 미만 모델로 교체되면 ORB 검증 층이 꺼진다.
// ---------------------------------------------------------------------------
namespace {

// 내용은 무작위, 행 수만 의미가 있는 ORB 디스크립터 (32바이트/행)
cv::Mat makeDescriptors(int rows) {
    cv::Mat d(rows, 32, CV_8UC1);
    cv::RNG rng(777);
    rng.fill(d, cv::RNG::UNIFORM, 0, 256);
    return d;
}

// 원본과 NCC는 거의 같지만(밝기 오프셋만 다름) 픽셀 값은 달라서
// "템플릿이 실제로 교체됐는지"를 구분할 수 있는 새 템플릿
cv::Mat makeShiftedTemplate(const cv::Mat& base) {
    cv::Mat t;
    base.convertTo(t, -1, 1.0, 10.0);
    return t;
}

bool sameMat(const cv::Mat& a, const cv::Mat& b) {
    return a.size() == b.size() && a.type() == b.type() && cv::norm(a, b, cv::NORM_INF) == 0;
}

class UpdateGateTest : public ::testing::TestWithParam<int> {};

}  // namespace

TEST_P(UpdateGateTest, RejectsWholeUpdateBelowMinDescriptors) {
    AcquisitionManager acq;
    acq.setTargetModel(makeTexturedObjectRoi());
    const cv::Mat oldTemplate = acq.getTargetTemplate().clone();
    const cv::Mat oldDesc     = acq.getTargetDescriptors().clone();
    ASSERT_GE(oldDesc.rows, AcquisitionManager::kMinUpdateDescriptors);

    const int n = GetParam();
    acq.updateTargetModel(makeShiftedTemplate(oldTemplate), makeDescriptors(n));

    EXPECT_TRUE(sameMat(acq.getTargetTemplate(), oldTemplate))   << "n=" << n << " 템플릿이 교체됨";
    EXPECT_TRUE(sameMat(acq.getTargetDescriptors(), oldDesc))    << "n=" << n << " 디스크립터가 교체됨";
}
INSTANTIATE_TEST_SUITE_P(BelowMin, UpdateGateTest, ::testing::Values(1, 6));

TEST(AcquisitionManagerTest, AcceptsUpdateAtMinDescriptors) {
    AcquisitionManager acq;
    acq.setTargetModel(makeTexturedObjectRoi());
    const cv::Mat oldTemplate = acq.getTargetTemplate().clone();
    const int n = AcquisitionManager::kMinUpdateDescriptors;   // 경계값 7은 승인

    acq.updateTargetModel(makeShiftedTemplate(oldTemplate), makeDescriptors(n));

    EXPECT_EQ(acq.getTargetDescriptors().rows, n);
    EXPECT_FALSE(sameMat(acq.getTargetTemplate(), oldTemplate)) << "템플릿이 교체되지 않음";
}

TEST(AcquisitionManagerTest, RejectsUpdateWithZeroDescriptors) {   // 기존 동작 유지 확인
    AcquisitionManager acq;
    acq.setTargetModel(makeTexturedObjectRoi());
    const cv::Mat oldDesc = acq.getTargetDescriptors().clone();

    acq.updateTargetModel(makeShiftedTemplate(acq.getTargetTemplate()), cv::Mat());

    EXPECT_TRUE(sameMat(acq.getTargetDescriptors(), oldDesc));
}
