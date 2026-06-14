# Office Attendance & Tracking System — Phase 3

Multi-camera employee tracking and attendance for 20–25 employees across 4 cameras.

---

## Architecture

```
Camera 1 (Entry)  ──┐
                     ├── camera_processor.py (InsightFace)
Camera 2 (Exit)   ──┘     → attendance_db.py

Camera 3 (Office A) ──┐
                       ├── office_tracker.py
Camera 4 (Office B) ──┘     → face_engine.py (InsightFace)
                               → reid_engine.py (OSNet ReID)
                               → tracker.py (ByteTrack + GlobalRegistry)
                               → activity_recognizer.py
                               → tracking_db.py
```

## New Files (Phase 3)

| File | Role |
|------|------|
| `face_engine.py` | InsightFace ArcFace wrapper — replaces MTCNN + DeepFace |
| `reid_engine.py` | OSNet cross-camera re-identification |
| `tracker.py` | ByteTrack per-camera + GlobalIdentityRegistry |
| `activity_recognizer.py` | Activity state classification per track |
| `office_tracker.py` | Multi-camera tracking engine (Camera 3 + 4) |
| `tracking_db.py` | New DB tables: presence, tracking_events, activity_log |

## Setup (first time)

```bash
python setup.py
```

## Enroll employees

```
enrollment/
  Harsha/
    photo1.jpg  photo2.jpg  photo3.jpg  photo4.jpg
  Ravi Kumar/
    photo1.jpg  photo2.jpg  …
```

```bash
python enroll_employees.py
```

## Configure cameras

Edit `config.py`:

```python
ENTRY_CAMERA_URL    = "rtsp://admin:pass@192.168.1.10/stream1"
EXIT_CAMERA_URL     = "rtsp://admin:pass@192.168.1.11/stream1"
OFFICE_CAMERA_A_URL = "rtsp://admin:pass@192.168.1.12/stream1"
OFFICE_CAMERA_B_URL = "rtsp://admin:pass@192.168.1.13/stream1"
```

## Run

```bash
python run.py
# Dashboard: http://localhost:5000
```

## Dashboard tabs

- **Live Tracking** — real-time presence: name, camera, activity, duration, verified status
- **Attendance** — entry/exit times, hours worked, session breakdown
- **Activity** — working/idle/active time per employee
- **Monthly** — attendance percentage reports
- **Security** — unknown person detections with face crops

## Identity resolution flow (Camera 3 & 4)

```
Frame → YOLO detect person → ByteTrack (local track_id)
         ↓ every 5 frames
    InsightFace on person crop
         ↓ match found?
    YES → assign emp_id (face_verified=True)
    NO  → OSNet ReID → search gallery
              ↓ sim > 0.65?
          YES → reuse existing global_id
          NO  → assign new Person_XXXX
```

## Activity states

| State | Trigger |
|-------|---------|
| `working_with_laptop` | Laptop/keyboard detected overlapping person box |
| `on_call` | Phone near upper body |
| `in_meeting` | Person in meeting zone |
| `idle` | No movement for 30+ frames |
| `active` | Moving, no specific work cues |

## Key config parameters

```python
FACE_DETECTION_INTERVAL  = 5      # run face recog every N frames
REID_SIMILARITY_THRESH   = 0.65   # ReID match threshold
IDLE_FRAME_COUNT         = 30     # frames before idle state
TRACKING_FRAME_INTERVAL_SECONDS = 1  # office camera speed
```
