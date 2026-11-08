#!/usr/bin/env python3
"""두 실행(A/B) 결과 비교 재생 뷰어 — GT 없이도 동작
사용: python3 scripts/compare_runs.py data/aircraft/move_v_fast_01.mp4 \
        --a base.csv --b patch.csv [--labels base,patch] [--gt data/aircraft/<clip>.gt_keys.csv]
  파랑 = A bbox, 초록 = B bbox, 자홍 = GT(선택)
  하단 타임라인 2줄(A 위, B 아래): 초록 TRACK / 빨강 LOST / 노랑 REACQUIRE / 회색 SEARCH / 검정 기록 없음
  키: space 일시정지/재생, d 다음 프레임, a 이전 프레임, n 다음 '상태 불일치' 프레임, q 종료
"""
import argparse, os
import cv2, numpy as np, pandas as pd

W, H = 640, 480
STRIP_H, GAP = 14, 4
COLOR_A, COLOR_B, COLOR_GT = (255, 100, 0), (0, 255, 0), (255, 0, 255)   # BGR
STATE_COLOR = {"TRACK": (80, 190, 80), "LOST": (60, 60, 220),
               "REACQUIRE": (0, 210, 255), "SEARCH": (150, 150, 150)}


def load_run(path):
    return pd.read_csv(path).set_index("frame")


def build_strip(run, n):
    """프레임 0..n-1을 가로 W픽셀에 매핑한 상태 색 막대 (STRIP_H × W)."""
    strip = np.zeros((STRIP_H, W, 3), np.uint8)
    for x in range(W):
        i = int(x * n / W)
        if i in run.index:
            strip[:, x] = STATE_COLOR.get(str(run.at[i, "fsm_state"]), (40, 40, 40))
    return strip


def state_of(run, i):
    return str(run.at[i, "fsm_state"]) if i in run.index else "-"


def next_diff(a, b, i, n):
    """i 다음부터 두 실행의 상태가 처음 달라지는 프레임 (없으면 i)."""
    for j in range(i + 1, n):
        if state_of(a, j) != state_of(b, j):
            return j
    return i


def draw_box(img, run, i, color):
    if i in run.index and run.at[i, "bbox_w"] > 0:
        x, y, w, h = (int(run.at[i, c]) for c in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def put(img, text, y, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def compose(frame, i, n, a, b, strips, labels, gt=None):
    img = cv2.resize(frame, (W, H))
    if gt is not None and i in gt.index and int(gt.at[i, "visible"]) == 1:
        x, y, w, h = gt.loc[i, ["x", "y", "w", "h"]].round().astype(int)
        cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_GT, 2)
    draw_box(img, a, i, COLOR_A)
    draw_box(img, b, i, COLOR_B)

    sa, sb = state_of(a, i), state_of(b, i)
    put(img, f"frame {i}/{n - 1}" + ("   << DIFF" if sa != sb else ""), 22,
        (0, 255, 255) if sa != sb else (255, 255, 255))
    put(img, f"A {labels[0]}: {sa}", 44, COLOR_A)
    put(img, f"B {labels[1]}: {sb}", 66, COLOR_B)

    # 하단 타임라인 + 현재 프레임 커서
    panel = np.zeros((2 * STRIP_H + 3 * GAP, W, 3), np.uint8)
    panel[GAP:GAP + STRIP_H] = strips[0]
    panel[2 * GAP + STRIP_H:2 * GAP + 2 * STRIP_H] = strips[1]
    cx = int(i * W / n)
    cv2.line(panel, (cx, 0), (cx, panel.shape[0] - 1), (255, 255, 255), 1)
    return np.vstack([img, panel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--a", required=True, help="실행 A 메트릭 CSV (예: 기준선)")
    ap.add_argument("--b", required=True, help="실행 B 메트릭 CSV (예: 패치)")
    ap.add_argument("--labels", default="A,B", help="표시 이름 2개, 쉼표 구분")
    ap.add_argument("--gt", help="GT 키프레임 CSV (선택)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--delay", type=int, default=33)
    args = ap.parse_args()

    a, b = load_run(args.a), load_run(args.b)
    labels = (args.labels.split(",") + ["A", "B"])[:2]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상 열기 실패: {args.video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    gt = None
    if args.gt:
        from eval_accuracy import interpolate_gt
        gt = interpolate_gt(pd.read_csv(args.gt), np.arange(n), "linear").set_index("frame")

    strips = (build_strip(a, n), build_strip(b, n))
    diff = sum(state_of(a, i) != state_of(b, i) for i in range(n))
    print(f"[compare] A={args.a}  B={args.b}  frames={n}  상태 불일치 {diff}프레임")

    i, paused, pos = max(0, args.start), False, -1
    while 0 <= i < n:
        if pos != i:                                   # 순차 재생이 아니면 탐색(seek)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read(); pos = i + 1
        if not ok:
            break
        cv2.imshow("compare_runs", compose(frame, i, n, a, b, strips, labels, gt))
        key = cv2.waitKey(0 if paused else args.delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            i += 0 if paused else 1
        elif key == ord("a"):
            paused, i = True, max(0, i - 1)
        elif key == ord("d"):
            paused, i = True, i + 1
        elif key == ord("n"):
            paused, i = True, next_diff(a, b, i, n)
        elif not paused:
            i += 1
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
