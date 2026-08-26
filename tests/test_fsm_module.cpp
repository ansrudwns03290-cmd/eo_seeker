#include <gtest/gtest.h>
#include "fsm/FsmModule.hpp"
#include "common/Frame.hpp"

namespace {

Frame makeFrame(int64_t timestamp_ms) {
    Frame f;
    f.is_valid = true;
    f.timestamp_ms = timestamp_ms;
    return f;
}

}  // namespace

TEST(FsmModuleTest, StartsInSearchState) {
    FsmModule fsm;
    EXPECT_EQ(fsm.getCurrentState(), FSMState::SEARCH);
    EXPECT_EQ(fsm.getStateString(), "SEARCH");
}

TEST(FsmModuleTest, ForceSetStateChangesState) {
    FsmModule fsm;
    fsm.forceSetState(FSMState::TRACK);
    EXPECT_EQ(fsm.getCurrentState(), FSMState::TRACK);
    EXPECT_EQ(fsm.getStateString(), "TRACK");
}

TEST(FsmModuleTest, TrackStaysTrackWithGoodConfidence) {
    FsmModule fsm;
    fsm.forceSetState(FSMState::TRACK);

    for (int i = 0; i < 10; ++i) {
        fsm.update(makeFrame(1000 + i * 33), 0.9f, true, cv::Point2f(0, 0));
    }
    EXPECT_EQ(fsm.getCurrentState(), FSMState::TRACK);
}

TEST(FsmModuleTest, TrackTransitionsToLostAfterFiveLowConfidenceFrames) {
    FsmModule fsm;
    fsm.forceSetState(FSMState::TRACK);

    for (int i = 0; i < 4; ++i) {
        fsm.update(makeFrame(1000 + i * 33), 0.1f, false, cv::Point2f(0, 0));
        EXPECT_EQ(fsm.getCurrentState(), FSMState::TRACK)
            << "4번째 저신뢰도 프레임까지는 아직 TRACK을 유지해야 함 (i=" << i << ")";
    }
    fsm.update(makeFrame(1132), 0.1f, false, cv::Point2f(0, 0));
    EXPECT_EQ(fsm.getCurrentState(), FSMState::LOST);
}

TEST(FsmModuleTest, ClassifiesHighSpeedLossWindowSize) {
    FsmModule fsm;
    fsm.forceSetState(FSMState::TRACK);

    for (int i = 0; i < 4; ++i) {
        fsm.update(makeFrame(1000 + i * 33), 0.1f, false, cv::Point2f(0, 0));
    }
    // LOST 진입 직전 프레임에 빠른 속도 벡터(크기 20 >= 임계값 15)를 전달
    fsm.update(makeFrame(1132), 0.1f, false, cv::Point2f(20.0f, 0.0f));
    ASSERT_EQ(fsm.getCurrentState(), FSMState::LOST);
    EXPECT_EQ(fsm.getSearchWindowSize(), 160);
}

TEST(FsmModuleTest, ClassifiesNormalSpeedLossWindowSize) {
    FsmModule fsm;
    fsm.forceSetState(FSMState::TRACK);

    for (int i = 0; i < 4; ++i) {
        fsm.update(makeFrame(1000 + i * 33), 0.1f, false, cv::Point2f(0, 0));
    }
    // 느린 속도 벡터(크기 5 < 임계값 15)
    fsm.update(makeFrame(1132), 0.1f, false, cv::Point2f(5.0f, 0.0f));
    ASSERT_EQ(fsm.getCurrentState(), FSMState::LOST);
    EXPECT_EQ(fsm.getSearchWindowSize(), 80);
}

TEST(FsmModuleTest, LostTransitionsToReacquireWhenCandidateFound) {
    FsmModule fsm;
    fsm.forceSetState(FSMState::TRACK);
    fsm.setTargetBox(cv::Rect(10, 10, 20, 20));

    for (int i = 0; i < 5; ++i) {
        fsm.update(makeFrame(1000 + i * 33), 0.1f, false, cv::Point2f(0, 0));
    }
    ASSERT_EQ(fsm.getCurrentState(), FSMState::LOST);

    // LOST 상태에서 후보가 발견되면(kcfSuccess=true) REACQUIRE로 전이
    fsm.update(makeFrame(1200), 0.0f, true, cv::Point2f(0, 0));
    EXPECT_EQ(fsm.getCurrentState(), FSMState::REACQUIRE);
}

TEST(FsmModuleTest, ReacquireSucceedsAndRestoresTargetBox) {
    FsmModule fsm;
    fsm.forceSetState(FSMState::TRACK);
    cv::Rect box(10, 10, 20, 20);
    fsm.setTargetBox(box);

    for (int i = 0; i < 5; ++i) {
        fsm.update(makeFrame(1000 + i * 33), 0.1f, false, cv::Point2f(0, 0));
    }
    fsm.update(makeFrame(1200), 0.0f, true, cv::Point2f(0, 0));  // -> REACQUIRE
    ASSERT_EQ(fsm.getCurrentState(), FSMState::REACQUIRE);

    fsm.update(makeFrame(1210), 0.75f, true, cv::Point2f(0, 0));  // 검증 통과(>=0.60)
    EXPECT_EQ(fsm.getCurrentState(), FSMState::TRACK);
    EXPECT_TRUE(fsm.getTargetBox() == box);
}

TEST(FsmModuleTest, ReacquireFailsReturnsToLost) {
    FsmModule fsm;
    fsm.forceSetState(FSMState::TRACK);
    fsm.setTargetBox(cv::Rect(10, 10, 20, 20));

    for (int i = 0; i < 5; ++i) {
        fsm.update(makeFrame(1000 + i * 33), 0.1f, false, cv::Point2f(0, 0));
    }
    fsm.update(makeFrame(1200), 0.0f, true, cv::Point2f(0, 0));  // -> REACQUIRE
    ASSERT_EQ(fsm.getCurrentState(), FSMState::REACQUIRE);

    fsm.update(makeFrame(1210), 0.30f, true, cv::Point2f(0, 0));  // 검증 실패(<0.60)
    EXPECT_EQ(fsm.getCurrentState(), FSMState::LOST);
}

TEST(FsmModuleTest, LostReturnsToSearchAfterGoldenTimeExpires) {
    FsmModule fsm;
    fsm.forceSetState(FSMState::TRACK);

    for (int i = 0; i < 5; ++i) {
        fsm.update(makeFrame(1000 + i * 33), 0.1f, false, cv::Point2f(0, 0));
    }
    ASSERT_EQ(fsm.getCurrentState(), FSMState::LOST);

    // 후보도 없이(kcfSuccess=false) 골든타임(1.5초)을 넘겨버리면 SEARCH로 복귀
    fsm.update(makeFrame(3000), 0.0f, false, cv::Point2f(0, 0));
    EXPECT_EQ(fsm.getCurrentState(), FSMState::SEARCH);
}
