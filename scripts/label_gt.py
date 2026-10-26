# scripts/label_gt.py  사용: python3 scripts/label_gt.py data/aircraft/move_h_fast_01.mp4 --step 5
import argparse, csv, cv2
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--step", type=int, default=10)
ap.add_argument("--frames", help="특정 프레임만 라벨링 (예: 45,82,122)")
a = ap.parse_args()

targets = set(map(int, a.frames.split(","))) if a.frames else None
suffix = ".gt_check.csv" if targets else ".gt_keys.csv"   # 검증용은 별도 파일 (키프레임 덮어쓰기 방지)

cap = cv2.VideoCapture(a.video); rows = []; idx = 0
while True:
    ok, f = cap.read()
    if not ok or (targets and idx > max(targets)): break
    if (idx in targets) if targets else (idx % a.step == 0):
        f = cv2.resize(f, (640, 480))                      # 런타임과 동일 좌표계
        cv2.putText(f, f"frame {idx}  drag+Enter=box  c=NO TARGET", (10, 20), 0, 0.5, (0,255,0), 1)
        x, y, w, h = cv2.selectROI("GT", f, False)
        rows.append([idx, x, y, w, h, int(w > 0)])         # c(취소) → visible=0
    idx += 1

with open(a.video.rsplit(".", 1)[0] + suffix, "w", newline="") as fp:
    csv.writer(fp, lineterminator="\n").writerows([["frame", "x", "y", "w", "h", "visible"]] + rows)
