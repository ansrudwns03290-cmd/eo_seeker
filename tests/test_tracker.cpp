#include <gtest/gtest.h>
#include "target_tracking/Tracker.hpp"

namespace {

// KCF(HOG 기반)가 잡을 만한 최소한의 경계를 제공하기 위해, 균일한 배경 위에
// 밝은 사각형 하나를 그린 합성 프레임을 만든다.
cv::Mat makeSyntheticFrame(const cv::Rect& targetBox, int width = 320, int height = 240) {
    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::rectangle(frame, targetBox, cv::Scalar(220, 220, 220), cv::FILLED);
    return frame;
}

}  // namespace

TEST(TrackerTest, InitFailsOnEmptyFrame) {
    Tracker tracker;
    cv::Mat empty;
    cv::Rect box(10, 10, 40, 40);
    EXPECT_FALSE(tracker.init(empty, box, cv::Mat(), cv::Mat()));
}

TEST(TrackerTest, InitFailsOnInvalidBbox) {
    Tracker tracker;
    cv::Rect box(130, 90, 60, 60);
    cv::Mat frame = makeSyntheticFrame(box);
    cv::Rect invalidBox(10, 10, 0, 0);  // width/height 0
    EXPECT_FALSE(tracker.init(frame, invalidBox, cv::Mat(), cv::Mat()));
}

TEST(TrackerTest, InitSucceedsWithValidTarget) {
    Tracker tracker;
    cv::Rect box(130, 90, 60, 60);
    cv::Mat frame = makeSyntheticFrame(box);
    EXPECT_TRUE(tracker.init(frame, box, cv::Mat(), cv::Mat()));
}

TEST(TrackerTest, UpdateFailsBeforeInit) {
    Tracker tracker;
    cv::Rect box(130, 90, 60, 60);
    cv::Mat frame = makeSyntheticFrame(box);

    Tracker::TrackingResult result = tracker.update(frame, cv::Mat(), cv::Mat());
    EXPECT_FALSE(result.success);
}

TEST(TrackerTest, UpdateTracksStationaryTarget) {
    Tracker tracker;
    cv::Rect box(130, 90, 60, 60);
    cv::Mat frame = makeSyntheticFrame(box);

    ASSERT_TRUE(tracker.init(frame, box, cv::Mat(), cv::Mat()));

    // 표적이 전혀 움직이지 않은 동일한 프레임을 다시 넣으면, KCF는 원래 위치를
    // 그대로 다시 찾아야 한다.
    Tracker::TrackingResult result = tracker.update(frame, cv::Mat(), cv::Mat());

    EXPECT_TRUE(result.success);

    cv::Point2f origCenter(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
    cv::Point2f newCenter(result.bbox.x + result.bbox.width / 2.0f,
                           result.bbox.y + result.bbox.height / 2.0f);
    double drift = cv::norm(newCenter - origCenter);

    // 정지된 표적이므로 중심 이동량이 크게 벗어나면 안 된다 (여유 있게 10px 허용).
    EXPECT_LT(drift, 10.0);
}
