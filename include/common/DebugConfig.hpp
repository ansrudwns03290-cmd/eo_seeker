#pragma once
#include <string>

// ============================================================
// [디버그 이미지 저장 스위치]
// 파이프라인 각 단계(초기 템플릿, ORB 특징점, LOST 후보 탐색, 전처리 결과 등)의
// 중간 이미지를 logs/debug/ 폴더에 파일로 남길지 여부를 한 곳에서 제어한다.
// 디버깅이 끝나면 kEnableImageDump만 false로 바꾸면 모든 imwrite 호출이 비활성화된다.
// ============================================================
namespace DebugConfig {
    constexpr bool kEnableImageDump = false;
    inline const std::string kDebugImageDir = "logs/debug/";
}
