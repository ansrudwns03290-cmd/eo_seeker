#include <gtest/gtest.h>
#include "target_tracking/KcfGeometry.hpp"

using namespace kcfgeom;

namespace {
bool isSmooth(int n) {
    for (int p : {2, 3, 5}) while (n % p == 0) n /= p;
    return n == 1;
}
cv::Point2f center(const cv::Rect& r) { return {r.x + r.width / 2.0f, r.y + r.height / 2.0f}; }
} // namespace

// --- computeScale: s = min(정책 배율, 48 / 긴 변) ---
TEST(KcfGeometry, ScaleCapsLargeTarget) {
    EXPECT_FLOAT_EQ(computeScale({196, 188}, false), 48.0f / 196.0f);
}
TEST(KcfGeometry, ScaleSmallModeUpscalesUpToTwo) {
    EXPECT_FLOAT_EQ(computeScale({20, 16}, true), 2.0f);        // 48/20=2.4 > 2 -> 2
    EXPECT_FLOAT_EQ(computeScale({40, 30}, true), 48.0f / 40.0f); // 상한이 우선
}
TEST(KcfGeometry, ScaleCapWinsForTallSmallTarget) {               // Q1: 50x150 -> s=2 금지
    EXPECT_FLOAT_EQ(computeScale({50, 150}, true), 48.0f / 150.0f);
}
TEST(KcfGeometry, ScaleNeverUpscalesInNormalMode) {
    EXPECT_FLOAT_EQ(computeScale({30, 30}, false), 1.0f);
}

// --- snapSmooth ---
TEST(KcfGeometry, SnapNearestSmoothWithinCap) {
    EXPECT_EQ(snapSmooth(47), 48);
    EXPECT_EQ(snapSmooth(37), 36);
    EXPECT_EQ(snapSmooth(13), 12);   // 12와 15 중 가까운 쪽
    EXPECT_EQ(snapSmooth(7), 8);     // 6과 8 동점 -> 큰 쪽
    EXPECT_EQ(snapSmooth(60), 48);   // 상한 초과 금지
    EXPECT_EQ(snapSmooth(0), 1);
}
TEST(KcfGeometry, SnapAlwaysSmoothForAllInputs) {
    for (int v = 1; v <= 200; ++v) {
        int s = snapSmooth(v);
        EXPECT_TRUE(isSmooth(s)) << v;
        EXPECT_LE(s, kMaxSide) << v;
    }
}

// --- toKcfBox: 크기는 2·3·5 곱, 중심 고정 ---
TEST(KcfGeometry, KcfBoxKeepsCenterAndSmoothSize) {
    cv::Rect orig(200, 150, 157, 191);                 // 실험 2의 소수 ROI
    float s = computeScale(orig.size(), false);
    cv::Size kcfFrame(static_cast<int>(640 * s), static_cast<int>(480 * s));
    cv::Rect k = toKcfBox(orig, s, kcfFrame);

    EXPECT_TRUE(isSmooth(k.width));
    EXPECT_TRUE(isSmooth(k.height));
    EXPECT_LE(std::max(k.width, k.height), kMaxSide);
    cv::Point2f c = center(orig) * s;
    EXPECT_NEAR(center(k).x, c.x, 0.5f);               // 반올림 오차 이내
    EXPECT_NEAR(center(k).y, c.y, 0.5f);
}
TEST(KcfGeometry, KcfBoxShiftedInsideFrameKeepsSize) {
    cv::Rect orig(-20, 0, 100, 100);                   // 왼쪽 경계 밖
    float s = computeScale(orig.size(), false);
    cv::Rect k = toKcfBox(orig, s, {static_cast<int>(640 * s), static_cast<int>(480 * s)});
    EXPECT_EQ(k.x, 0);
    EXPECT_TRUE(isSmooth(k.width));
}

// --- toOrigBox: 중심 /s, 크기는 원 ROI 고정 ---
TEST(KcfGeometry, RoundTripCenterErrorBoundedByScale) {
    cv::Rect orig(300, 200, 130, 153);
    float s = computeScale(orig.size(), false);
    cv::Rect k = toKcfBox(orig, s, {static_cast<int>(640 * s), static_cast<int>(480 * s)});
    cv::Rect back = toOrigBox(k, s, orig.size());

    EXPECT_EQ(back.size(), orig.size());
    // KCF 좌표 반올림 0.5px -> 원본에서 최대 0.5/s px (+출력 반올림 0.5px)
    float tol = 0.5f / s + 0.5f;
    EXPECT_NEAR(center(back).x, center(orig).x, tol);
    EXPECT_NEAR(center(back).y, center(orig).y, tol);
}
