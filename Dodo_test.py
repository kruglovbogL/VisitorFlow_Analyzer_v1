import argparse
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# --- ПАРАМЕТРЫ ---
PERSON_CLASS_ID = 0  # COCO: person
IOU_THRESHOLD = 0.1  # пересечение bbox человека с ROI стола
CONF_THRESHOLD = 0.4

STATE_EMPTY = "EMPTY"
STATE_OCCUPIED = "OCCUPIED"

def boxes_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video, e.g. video1.mp4")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Path to output video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model weights")
    parser.add_argument("--report", type=str, default="report.txt",
                        help="Path to text report")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # --- Выбор ROI стола на первом кадре ---
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame from video")

    roi = cv2.selectROI("Select table ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select table ROI")
    x, y, w, h = roi
    table_box = (x, y, x + w, y + h)

    # --- Модель YOLO ---
    model = YOLO(args.model)  # скачает веса при первом запуске [web:15][web:18]

    # --- Состояния и лог ---
    current_state = STATE_EMPTY
    last_state_change_ts = 0.0
    last_empty_ts = None
    pending_empty_ts = None  # время, когда стол стал пустым, в ожидании подхода
    event_rows = []
    delays_after_empty = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps
        frame_idx += 1

        # --- Детекция людей YOLO ---
        results = model(frame, verbose=False)[0]
        person_in_roi = False

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, classes, confs):
                if int(cls_id) != PERSON_CLASS_ID:
                    continue
                if conf < CONF_THRESHOLD:
                    continue

                px1, py1, px2, py2 = box
                person_box = (px1, py1, px2, py2)
                iou = boxes_iou(person_box, table_box)
                if iou > IOU_THRESHOLD:
                    person_in_roi = True
                    break

        new_state = STATE_OCCUPIED if person_in_roi else STATE_EMPTY

        # --- Обработка событий ---
        event_type = None

        if new_state != current_state:
            # смена состояния стола
            if current_state == STATE_OCCUPIED and new_state == STATE_EMPTY:
                # гость(и) ушли -> стол пустой
                pending_empty_ts = timestamp_sec
                event_type = "BECAME_EMPTY"
            elif current_state == STATE_EMPTY and new_state == STATE_OCCUPIED:
                # подход к столу после пустоты
                if pending_empty_ts is not None:
                    delay = timestamp_sec - pending_empty_ts
                    delays_after_empty.append(delay)
                    event_type = "APPROACH_AFTER_EMPTY"
                    pending_empty_ts = None
                else:
                    event_type = "BECAME_OCCUPIED"
            else:
                event_type = "STATE_CHANGE"

            current_state = new_state
            last_state_change_ts = timestamp_sec

        # --- Логируем каждую смену состояния и особые события ---
        if event_type is not None:
            event_rows.append(
                {
                    "timestamp_sec": timestamp_sec,
                    "state": current_state,
                    "event_type": event_type,
                }
            )

        # --- Рисуем ROI и состояние ---
        color = (0, 255, 0) if current_state == STATE_EMPTY else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        text_state = f"{current_state}"
        cv2.putText(
            frame,
            text_state,
            (x, y - 10 if y - 10 > 20 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        # Дополнительная инфа: время и возможная средняя задержка
        cv2.putText(
            frame,
            f"time: {timestamp_sec:.1f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if delays_after_empty:
            avg_delay = float(np.mean(delays_after_empty))
            cv2.putText(
                frame,
                f"avg delay: {avg_delay:.1f}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        out.write(frame)

        # Если хотите показывать во время обработки:
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- Аналитика и сохранение ---
    df_events = pd.DataFrame(event_rows)
    df_events.to_csv("events.csv", index=False)

    if delays_after_empty:
        avg_delay = float(np.mean(delays_after_empty))
    else:
        avg_delay = None

    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"Video: {args.video}")
        f.write(f"Table ROI: x={x}, y={y}, w={w}, h={h}")
        f.write(f"Events logged: {len(event_rows)}")
        if avg_delay is not None:
            f.write(f"Average delay after table becomes empty: {avg_delay:.2f} seconds")
        else:
            f.write("Average delay after table becomes empty: not enough events")
        f.write(f"Generated at: {datetime.now().isoformat()}")

    print("Done. Saved:", args.output)
    print("Events CSV: events.csv")
    print("Report:", args.report)


if __name__ == "__main__":
    main()