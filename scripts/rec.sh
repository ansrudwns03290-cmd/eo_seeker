#!/bin/bash
# 사용법: ./scripts/rec.sh speed_horizontal_fast_02 12
set -e
NAME=$1; SEC=$2
[ -z "$NAME" ] || [ -z "$SEC" ] && { echo "사용법: $0 <이름> <초>"; exit 1; }
CAT=${NAME%%_*}
OUT=~/clips/$CAT; mkdir -p "$OUT"
[ -e "$OUT/$NAME.avi" ] && { echo "이미 존재: $NAME"; exit 1; }

rpicam-vid -n -t $(( (SEC+3)*1000 )) \
  --mode 1536:864:10:P \
  --width 640 --height 480 --framerate 30 \
  --codec mjpeg -q 95 \
  -o "$OUT/$NAME.mjpeg" --save-pts "$OUT/$NAME.pts.txt"

ffmpeg -loglevel error -framerate 30 -i "$OUT/$NAME.mjpeg" -c copy "$OUT/$NAME.avi"
rm "$OUT/$NAME.mjpeg"

DROPS=$(awk 'NR>1{ if(p!="") { if($1-p>40) c++ } p=$1 } END{ print c+0 }' "$OUT/$NAME.pts.txt")
echo "완료: $OUT/$NAME.avi | drops=$DROPS"
[ "$DROPS" -gt 0 ] && echo "⚠ 드롭 발생 → 재촬영 권장"
exit 0
