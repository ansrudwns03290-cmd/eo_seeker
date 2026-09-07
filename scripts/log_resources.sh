#!/usr/bin/env bash
# Pi 리소스/온도 병행 로거
#
# eo_seeker 실행 중 별도 터미널에서 실행해서 CPU 사용률/메모리/온도/스로틀링 여부를
# 일정 간격으로 CSV에 기록한다. main.cpp의 프레임별 타이머(src/main.cpp의 metrics_csv)와는
# 완전히 독립적으로 동작하므로, 같은 세션을 두 CSV(프레임 타이밍 + 리소스)로 함께
# 분석하려면 실행 시각을 맞춰서 켜두면 된다.
#
# 사용법:
#   ./scripts/log_resources.sh [출력파일] [간격초]
#   기본값: logs/resource_<timestamp>.csv, 1초 간격
#
# 종료: Ctrl+C

set -euo pipefail

OUT_FILE="${1:-logs/resource_$(date +%Y%m%d_%H%M%S).csv}"
INTERVAL="${2:-1}"

mkdir -p "$(dirname "$OUT_FILE")"

echo "timestamp,temp_c,throttled_raw,eo_seeker_cpu_pct,eo_seeker_mem_pct" > "$OUT_FILE"
echo "[log_resources] 기록 시작 -> $OUT_FILE (간격: ${INTERVAL}s, Ctrl+C로 종료)"

while true; do
    ts=$(date +%s.%N)

    # vcgencmd는 Raspberry Pi OS 전용. 없으면 온도/스로틀링은 빈 값으로 남긴다.
    if command -v vcgencmd >/dev/null 2>&1; then
        temp=$(vcgencmd measure_temp | sed -E 's/temp=([0-9.]+).*/\1/')
        throttled=$(vcgencmd get_throttled | sed -E 's/throttled=//')
    else
        temp=""
        throttled=""
    fi

    # eo_seeker 프로세스의 CPU%/MEM% (여러 인스턴스가 떠 있으면 첫 줄만 사용).
    # 프로세스가 아직 안 떴거나 이미 종료됐으면 빈 값으로 남긴다.
    top_line=$(top -bn1 | grep -m1 '[e]o_seeker' || true)
    cpu_pct=$(echo "$top_line" | awk '{print $9}')
    mem_pct=$(echo "$top_line" | awk '{print $10}')

    echo "${ts},${temp},${throttled},${cpu_pct:-},${mem_pct:-}" >> "$OUT_FILE"
    sleep "$INTERVAL"
done
