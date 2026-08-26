#include <gtest/gtest.h>
#include "acquisition/AcquisitionManager.hpp"

namespace {

// ORB가 잡을 만한 코너/질감을 만들기 위한 합성 체커보드 ROI.
cv::Mat makeCheckerboardRoi(int size = 80, int cell = 10) {
    cv::Mat img(size, size, CV_8UC3, cv::Scalar(50, 50, 50));
    for (int y = 0; y < size; y += cell) {
        for (int x = 0; x < size; x += cell) {
            if (((x / cell) + (y / cell)) % 2 == 0) {
                cv::rectangle(img, cv::Rect(x, y, cell, cell), cv::Scalar(220, 220, 220), cv::FILLED);
            }
        }
    }
    return img;
}

}  // namespace

TEST(AcquisitionManagerTest, SetTargetModelExtractsDescriptorsFromTexturedImage) {
    AcquisitionManager acq;
    cv::Mat roi = makeCheckerboardRoi();

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
    cv::Mat roi = makeCheckerboardRoi();
    acq.setTargetModel(roi);

    cv::Mat gray(240, 320, CV_8UC1, cv::Scalar(50));
    cv::Rect invalidRect(0, 0, 0, 0);  // width/height 0
    cv::Rect outBox;

    EXPECT_FALSE(acq.detectCandidateInPredictArea(gray, invalidRect, 80, outBox));
}
