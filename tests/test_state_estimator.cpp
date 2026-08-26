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

    // 칼만필터는 속도를 직접 측정하지 않고, 반복되는 predict+correct 사이클을 거치며
    // 위치-속도 공분산이 쌓여야 비로소 속도를 추정할 수 있다. 10스텝(1초)만으로는
    // 부족해서(같은 필터를 파이썬으로 재현한 결과 10스텝→약 51.5, 30스텝→약 93.3,
    // 50스텝→약 99.1로 수렴하는 것을 확인) 50스텝까지 충분히 돌린 뒤 검증한다.
    for (int i = 1; i <= 50; ++i) {
        estimator.predict(0.1);
        estimator.update(cv::Point2f(10.0f * i, 0.0f));
    }

    cv::Point2f vel = estimator.getEstimatedVelocity();
    EXPECT_NEAR(vel.x, 100.0f, 15.0f);  // 50스텝 기준 이론상 ~99.1로 수렴 (여유 있게 15 허용)
}

TEST(StateEstimatorTest, ResetClearsInitializedFlag) {
    StateEstimator estimator;
    estimator.initialize(cv::Point2f(50.0f, 50.0f));
    ASSERT_TRUE(estimator.isInitialized());

    estimator.reset();
    EXPECT_FALSE(estimator.isInitialized());
}
