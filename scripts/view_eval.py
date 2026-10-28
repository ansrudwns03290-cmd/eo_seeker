#!/usr/bin/env python3
"""GT vs 추적기 결과 재생 뷰어
사용: python3 scripts/view_eval.py <clip> --commit 9c55168 [--run 51] [--start 120] [--delay 33]
  red = GT(보간), green = 추적기 bbox, 노랑 글자 = 전환 구간(평가 제외)
  키: space 일시정지/재생, d 다음 프레임, a 이전 프레임, q 종료
"""
import argparse, glob, os
import cv2, numpy as np, pandas as pd
from eval_accuracy import interpolate_gt

ap = argparse.ArgumentParser()
ap.add_argument("clip"); ap.add_argument("--commit", required=True)
ap.add_argument("--run", type=int, help="생략하면 해당 커밋의 첫 실행")
ap.add_argument("--start", type=int, default=0); ap.add_argument("--delay", type=int, default=33)
ap.add_argument("--gt-dir", default="data/aircraft"); ap.add_argument("--logs", default="logs")
a = ap.parse_args()

runs = sorted(glob.glob(os.path.join(a.logs, f"{a.commit}_*_{a.clip}.csv")))
if a.run is not None:
    runs = [f for f in runs if int(os.path.basename(f).split("_")[1]) == a.run]
if not runs:
    raise SystemExit(f"메트릭 CSV 없음: {a.logs}/{a.commit}_*_{a.clip}.csv")
pred = pd.read_csv(runs[0]).set_index("frame")
keys = pd.read_csv(os.path.join(a.gt_dir, f"{a.clip}.gt_keys.csv"))

cap = cv2.VideoCapture(os.path.join(a.gt_dir, f"{a.clip}.mp4"))
if not cap.isOpened():
    raise SystemExit("영상 열기 실패")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
gt = interpolate_gt(keys, np.arange(n), "linear").set_index("frame")
print(f"[view] {runs[0]}  frames={n}")


def draw(i, frame):
    img = cv2.resize(frame, (640, 480))
    vis = int(gt.loc[i, "visible"]) if i in gt.index else None
    if vis == 1:
        x, y, w, h = gt.loc[i, ["x", "y", "w", "h"]].round().astype(int)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0,  255), 2)
    state, cle = "-", None
    if i in pred.index:
        r = pred.loc[i]
        state = str(r["fsm_state"])
        if r["bbox_w"] > 0:
            px, py, pw, ph = int(r["bbox_x"]), int(r["bbox_y"]), int(r["bbox_w"]), int(r["bbox_h"])
            cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
            if vis == 1:
                g = gt.loc[i]
                cle = np.hypot(g.x + g.w / 2 - (px + pw / 2), g.y + g.h / 2 - (py + ph / 2))
    gt_txt = {1: "GT visible", 0: "GT none", -1: "GT transition(ignored)", None: "GT -"}[vis]
    color = (0, 255, 255) if vis == -1 else (255, 255, 255)
    lines = [f"frame {i}  state {state}", gt_txt,
             f"CLE {cle:.1f}px" if cle is not None else ("CLE inf (no box)" if vis == 1 else "CLE -")]
    for k, t in enumerate(lines):
        cv2.putText(img, t, (10, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, t, (10, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return img


i, paused, pos = max(0, a.start), False, -1
while 0 <= i < n:
    if pos != i:                                   # 순차 재생이 아니면 탐색(seek)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read(); pos = i + 1
    if not ok:
        break
    cv2.imshow("view_eval", draw(i, frame))
    key = cv2.waitKey(0 if paused else a.delay) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        paused = not paused
        i += 0 if paused else 1
    elif key == ord("a"):
        paused, i = True, max(0, i - 1)
    elif key == ord("d"):
        paused, i = True, i + 1
    elif not paused:
        i += 1
cap.release(); cv2.destroyAllWindows()
