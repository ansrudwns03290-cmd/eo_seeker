#include <gtest/gtest.h>
#include "state_estimation/StateEstimator.hpp"

TEST(StateEstimatorTest, InitializeSetsPosition) {
    StateEstimator estimator;
    cv::Point2f start(100.0f, 200.0f);
    estimator.initialize(start);

    EXPECT_TRUE(estimator.isInitialized());
    cv::Point2f pos = estimator.getEstimatedPosition();
    EXPECT_NEAR(pos.x, 100.0f, 1e-3);
    EXPECT_NEAR(pos.y, 200.0f, 1e-3);
}

TEST(StateEstimatorTest, PredictFollowsConstantVelocity) {
    StateEstimator estimator;
    estimator.initialize(cv::Point2f(0.0f, 0.0f));

    // 초당 100px로 등속 이동하는 표적을 흉내낸 측정값을 반복 입력해서
    // 칼만필터의 속도 추정치가 그 값으로 수렴하는지 확인한다.
    for (int i = 1; i <= 10; ++i) {
        estimator.predict(0.1);
        estimator.update(cv::Point2f(10.0f * i, 0.0f));
    }

    cv::Point2f vel = estimator.getEstimatedVelocity();
    EXPECT_NEAR(vel.x, 100.0f, 20.0f);  // 칼만필터 수렴 오차를 감안한 허용 범위
}

TEST(StateEstimatorTest, ResetClearsInitializedFlag) {
    StateEstimator estimator;
    estimator.initialize(cv::Point2f(50.0f, 50.0f));
    ASSERT_TRUE(estimator.isInitialized());

    estimator.reset();
    EXPECT_FALSE(estimator.isInitialized());
}
