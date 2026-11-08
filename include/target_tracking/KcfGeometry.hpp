#pragma once

#include <opencv2/core.hpp>

/**
 * @brief KCF 입력 좌표계 <-> 원본 좌표계 변환 (EO-33 패치 상한 + EO-5 DFT 친화 스냅)
 *
 * 배율 s = KCF 입력 좌표 / 원본 좌표. (s<1 축소, s=1 그대로, s>1 확대)
 * TrackerKCF::Params.resize=false 전제에서 KCF 내부 FFT 패치 = 2 × KCF 입력 bbox.
 * 프레임 없이 계산만 하는 순수 함수라 GoogleTest로 단독 검증한다.
 */
namespace kcfgeom {

// [EO-33 11/08 재설계] 강건성 우선: 큰 표적은 실효 해상도 0.5를 하한으로 두고,
// 아주 큰 표적만 긴 변 상한으로 추가 축소해 처리시간을 보호한다.
// (상한 48 단독 설계는 해상도 0.27~0.33에서 추적 강건성이 무너져 폐기 — EO-33 진행 기록)
constexpr int   kMaxSide    = 96;       // KCF 입력 bbox 긴 변 상한 (FFT 패치 긴 변 192px)
constexpr int   kLargeArea  = 80 * 80;  // 이 면적을 넘는 표적은 0.5 배율 (기존 KCF 내부 max_patch_size와 동일 기준)
constexpr float kLargeScale = 0.5f;

/**
 * @brief 배율 결정. s = min(정책 배율, maxSide / 원 ROI 긴 변)
 *        정책 배율: 소형 모드 2 / 면적 > kLargeArea 이면 0.5 / 그 외 1. 상한이 항상 우선.
 */
float computeScale(cv::Size orig, bool smallMode, int maxSide = kMaxSide);

/**
 * @brief v에 가장 가까운 2·3·5의 곱 (1 이상, maxV 이하). 거리가 같으면 큰 쪽.
 */
int snapSmooth(int v, int maxV = kMaxSide);

/**
 * @brief 원본 bbox -> KCF 입력 bbox. 축소/확대 -> 크기 스냅 -> 중심 고정 -> 프레임 안으로 이동.
 * 경계에서는 크기를 자르지 않고 위치만 옮긴다(스냅한 크기 유지).
 * @param kcfFrame 배율 적용 후 KCF에 들어가는 프레임 크기
 */
cv::Rect toKcfBox(const cv::Rect& orig, float s, cv::Size kcfFrame, int maxSide = kMaxSide);

/**
 * @brief KCF 출력 bbox -> 원본 bbox. 중심만 /s 후 반올림하고, 크기는 origSize로 고정.
 */
cv::Rect toOrigBox(const cv::Rect& kcfBox, float s, cv::Size origSize);

} // namespace kcfgeom
