#include "target_tracking/KcfGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace kcfgeom {

float computeScale(cv::Size orig, bool smallMode, int maxSide) {
    const int longSide = std::max(orig.width, orig.height);
    if (longSide <= 0) return 1.0f;

    const float policy = smallMode ? 2.0f : 1.0f;
    const float cap = static_cast<float>(maxSide) / static_cast<float>(longSide);
    return std::min(policy, cap);
}

namespace {
bool isSmooth(int n) {
    for (int p : {2, 3, 5}) {
        while (n % p == 0) n /= p;
    }
    return n == 1;
}
} // namespace

int snapSmooth(int v, int maxV) {
    v = std::max(v, 1);
    int best = 1;
    for (int c = 1; c <= maxV; ++c) {
        if (!isSmooth(c)) continue;
        // 거리가 같거나 더 가까우면 갱신 -> 동점이면 나중에 나온(큰) 값
        if (std::abs(c - v) <= std::abs(best - v)) best = c;
    }
    return best;
}

cv::Rect toKcfBox(const cv::Rect& orig, float s, cv::Size kcfFrame, int maxSide) {
    const float cx = (orig.x + orig.width / 2.0f) * s;
    const float cy = (orig.y + orig.height / 2.0f) * s;

    int w = snapSmooth(static_cast<int>(std::lround(orig.width * s)), maxSide);
    int h = snapSmooth(static_cast<int>(std::lround(orig.height * s)), maxSide);
    w = std::min(w, kcfFrame.width);
    h = std::min(h, kcfFrame.height);

    int x = static_cast<int>(std::lround(cx - w / 2.0f));
    int y = static_cast<int>(std::lround(cy - h / 2.0f));
    x = std::clamp(x, 0, kcfFrame.width - w);
    y = std::clamp(y, 0, kcfFrame.height - h);

    return cv::Rect(x, y, w, h);
}

cv::Rect toOrigBox(const cv::Rect& kcfBox, float s, cv::Size origSize) {
    const float cx = (kcfBox.x + kcfBox.width / 2.0f) / s;
    const float cy = (kcfBox.y + kcfBox.height / 2.0f) / s;
    const int x = static_cast<int>(std::lround(cx - origSize.width / 2.0f));
    const int y = static_cast<int>(std::lround(cy - origSize.height / 2.0f));
    return cv::Rect(x, y, origSize.width, origSize.height);
}

} // namespace kcfgeom
