#include <gtest/gtest.h>
#include "acquisition/AcquisitionManager.hpp"

namespace {

// setTargetModel()은 ROI 전체가 아니라 Otsu 이진화 + findContours로 찾은
// "가장 큰 물체" bbox만 잘라서 그 위에 ORB를 돌린다. ORB는 edgeThreshold=31이라
// 잘라낸 물체가 최소 62px(2*edgeThreshold) 이상이어야 코너를 하나라도 찾는다.
// 그래서 배경(균일한 어두운 색)과 뚜렷이 구분되는 objectSize(기본 90px)짜리
// 물체를 하나 두고, 그 물체 내부에만 체커보드 질감을 넣어 두 조건을 동시에
// 만족시킨다: (1) Otsu/컨투어가 물체 전체를 하나의 큰 블록으로 인식,
// (2) 그 블록 내부에 ORB가 잡을 코너가 충분히 존재.
cv::Mat makeTexturedObjectRoi(int canvasSize = 150, int objectSize = 90, int cell = 10) {
    cv::Mat img(canvasSize, canvasSize, CV_8UC3, cv::Scalar(50, 50, 50));

    int offset = (canvasSize - objectSize) / 2;
    for (int y = 0; y < objectSize; y += cell) {
        for (int x = 0; x < objectSize; x += cell) {
            // 두 체커 색(200/160) 모두 배경(50)과는 확실히 구분되면서 서로도
            // 구분되는 밝기라, Otsu 이진화에서는 하나의 전경 덩어리로 묶이고
            // ORB에서는 각 셀 경계가 코너로 잡힌다.
            cv::Scalar color = (((x / cell) + (y / cell)) % 2 == 0)
                                    ? cv::Scalar(200, 200, 200)
                                    : cv::Scalar(160, 160, 160);
            cv::rectangle(img, cv::Rect(offset + x, offset + y, cell, cell), color, cv::FILLED);
        }
    }
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
