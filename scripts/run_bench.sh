#!/usr/bin/env bash
# 벤치마크 실행 프로토콜 (#7)
# 사용법: ./scripts/run_bench.sh [반복횟수=5]
# 전제: 작업 트리 clean(커밋 해시=측정 코드), pi-release 빌드 완료, 영상별 .roi 존재
set -euo pipefail
cd "$(dirname "$0")/.."

REPS="${1:-5}"
BIN="build/pi-release/eo_seeker"
LIST="scripts/bench_list.txt"
COOLDOWN="${COOLDOWN:-5}"

# --- 0) 사전 점검 ---
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[FAIL] 커밋되지 않은 변경이 있음. 결과의 커밋 해시와 코드가 달라지므로 커밋 후 실행"; exit 1
fi
HASH=$(git rev-parse --short HEAD)
cmake --build --preset pi-release -j4 > /dev/null || { echo "[FAIL] 빌드 실패"; exit 1; }

[[ -x "$BIN" ]] || { echo "[FAIL] $BIN 없음 (cmake --build --preset pi-release)"; exit 1; }
grep -q 'CMAKE_BUILD_TYPE:STRING=Release' build/pi-release/CMakeCache.txt \
  || { echo "[FAIL] Release 빌드 아님"; exit 1; }

mapfile -t VIDEOS < <(grep -v '^\s*#' "$LIST" | sed '/^\s*$/d')
for v in "${VIDEOS[@]}"; do
  [[ -f "$v" ]] || { echo "[FAIL] 영상 없음: $v"; exit 1; }
  [[ -f "${v%.*}.roi" ]] || { echo "[FAIL] ROI 프리셋 없음: ${v%.*}.roi (헤드리스에서 선택창 불가)"; exit 1; }
done

# --- 1) 환경 고정 ---
GOV=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
if [[ "$(cat $GOV)" != "performance" ]]; then
  echo "[Setup] CPU governor → performance (sudo)"
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
fi
[[ "$(cat $GOV)" == "performance" ]] || { echo "[FAIL] governor 설정 실패"; exit 1; }
export EO_SEEKER_NO_DISPLAY=1

OUT="results/bench_${HASH}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/csv"
{ echo "commit=$HASH reps=$REPS date=$(date -Is)"; uname -a
  echo "governor=$(cat $GOV)"; vcgencmd get_throttled 2>/dev/null || true; } > "$OUT/env.txt"

./scripts/log_resources.sh "$OUT/resources.csv" 1 > /dev/null &
RES_PID=$!
trap 'kill $RES_PID 2>/dev/null || true' EXIT

# --- 2) 실행 + 무결성 검증 ---
FAIL=0
for v in "${VIDEOS[@]}"; do
  name=$(basename "${v%.*}")
  frames=$(ffprobe -v error -count_frames -select_streams v:0 \
           -show_entries stream=nb_read_frames -of csv=p=0 "$v")
  expected=$((frames - 1))   # 프레임 0(ROI), 1(워밍업 직후) 미기록 + 헤더 1줄
  for r in $(seq 1 "$REPS"); do
    "$BIN" "$v" > /dev/null 2>&1 || { echo "[FAIL] $name run$r 비정상 종료"; FAIL=1; continue; }
    csv=$(ls -t logs/${HASH}_*_${name}.csv | head -1)
    lines=$(wc -l < "$csv")
    if [[ "$lines" -ne "$expected" ]]; then
      echo "[FAIL] $name run$r: csv_lines=$lines expected=$expected"; FAIL=1
    else
      echo "[OK]   $name run$r ($lines lines)"
    fi
    cp "$csv" "$OUT/csv/"
    sleep "$COOLDOWN"
  done
done

echo "after: $(vcgencmd get_throttled 2>/dev/null || echo n/a)" >> "$OUT/env.txt"

# --- 3) 재집계 + 반복 편차 ---
python3 scripts/aggregate_metrics.py "$OUT/csv" --out "$OUT"
python3 - "$OUT/runs.csv" <<'EOF' | tee "$OUT/p95_spread.txt"
import sys, pandas as pd
r = pd.read_csv(sys.argv[1]); r = r[r.scope == "TRACK"]
g = r.groupby("video")["total_ms_p95"].agg(["mean", "std", "min", "max"])
g["cv_pct"] = 100 * g["std"] / g["mean"]
print(g.round(2).to_string())
EOF

echo "[Done] 결과: $OUT"

mkdir -p "$OUT/csv"
exec > >(tee -a "$OUT/run.log") 2>&1   # ← 추가: 콘솔 출력 전체를 run.log에도 기록

exit $FAIL
