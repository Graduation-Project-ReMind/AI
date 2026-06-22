import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import deque



@dataclass
class GazeFrame:
    """Gaze data captured for a single video frame."""
    timestamp: float
    gaze_direction: str          
    left_ratio: float             
    right_ratio: float            
    vertical_ratio: float        
    blink_detected: bool
    attention_score: float        
    ear: float = 0.0             


@dataclass
class AttentionMetrics:
    total_frames: int = 0
    fixation_count: int = 0
    saccade_count: int = 0
    total_fixation_duration: float = 0.0
    total_distraction_duration: float = 0.0
    blink_count: int = 0
    direction_counts: dict = field(default_factory=lambda: {
        "center": 0, "left": 0, "right": 0,
        "up": 0, "down": 0, "blink": 0, "unknown": 0
    })


    @property
    def attention_percentage(self) -> float:
        total = self.total_fixation_duration + self.total_distraction_duration
        return (self.total_fixation_duration / total * 100) if total > 0 else 0.0
    ##########################################################################

    @property
    def avg_fixation_duration(self) -> float:
        return (self.total_fixation_duration / self.fixation_count
                if self.fixation_count > 0 else 0.0)
    ##########################################################################

    @property
    def saccade_rate(self) -> float:
        total_time = self.total_fixation_duration + self.total_distraction_duration
        return self.saccade_count / max(total_time, 1.0)
    ##########################################################################

    @property
    def blink_rate_per_min(self) -> float:
        total_time = self.total_fixation_duration + self.total_distraction_duration
        return (self.blink_count / max(total_time, 1.0)) * 60.0
    ##########################################################################



LEFT_EYE_INDICES  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                        388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                        157, 158, 159, 160, 161, 246]

LEFT_IRIS_INDICES  = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]

LEFT_EYE_LEFT_CORNER  = 362
LEFT_EYE_RIGHT_CORNER = 263
LEFT_EYE_TOP          = 386
LEFT_EYE_BOTTOM       = 374

RIGHT_EYE_LEFT_CORNER  = 33
RIGHT_EYE_RIGHT_CORNER = 133
RIGHT_EYE_TOP          = 159
RIGHT_EYE_BOTTOM       = 145



class GazeTracker:

    HORIZONTAL_LEFT_THRESHOLD  = 0.38    # كانت 0.42 — وسّعنا الـ center zone
    HORIZONTAL_RIGHT_THRESHOLD = 0.62    # كانت 0.58
    VERTICAL_UP_THRESHOLD      = 0.35    # كانت 0.38
    VERTICAL_DOWN_THRESHOLD    = 0.65    # كانت 0.62

    EAR_BLINK_THRESHOLD   = 0.42   # معايَر: open≈0.50 / blink_min≈0.37 → نص الفرق
    EAR_BLINK_FRAMES      = 2      # لازم يفضل مسكر فريمين متتاليين
    BLINK_COOLDOWN_S      = 0.20

    ATTENTION_WINDOW_FRAMES = 90    

    MIN_FIXATION_DURATION_S = 0.15   

    W_CENTERED  = 0.50   # الوزن الأكبر للنظر للشاشة
    W_FIXATION  = 0.25   # ثبات النظر مهم
    W_SACCADE   = 0.15   # عقوبة الحركة السريعة
    W_ENTROPY   = 0.10   # عقوبة التشتت

    SACCADE_K             = 5.0     # خفضنا من 8.0 → العقوبة تبدأ بدري أكتر
    MAX_SACCADE_PENALTY   = 0.50    # رفعنا الحد الأقصى من 0.35

    def __init__(self,
                min_detection_confidence: float = 0.7,
                min_tracking_confidence:  float = 0.7):

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.gaze_history:    deque = deque(maxlen=self.ATTENTION_WINDOW_FRAMES)
        self.dir_history:     deque = deque(maxlen=self.ATTENTION_WINDOW_FRAMES)
        self.fixation_history: deque = deque(maxlen=self.ATTENTION_WINDOW_FRAMES)
        self.metrics       = AttentionMetrics()
        self.session_data: List[GazeFrame] = []

        self._prev_direction: str   = "unknown"
        self._fixation_start: float = time.time()
        self._in_fixation:    bool  = False
        self._current_fixation_dir: str = "unknown"
        self._blink_cooldown: float = 0.0
        self._blink_frame_count: int = 0
        self._dir_buffer: deque = deque(maxlen=3)
        self._stable_direction: str = "unknown"

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[GazeFrame]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True
        annotated = frame.copy()

        if not results.multi_face_landmarks:
            self._draw_no_face_overlay(annotated)
            return annotated, None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        left_ic  = self._iris_center(landmarks, LEFT_IRIS_INDICES,  w, h)
        right_ic = self._iris_center(landmarks, RIGHT_IRIS_INDICES, w, h)

        left_corners  = self._eye_corners(landmarks,
                                            LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER,
                                            LEFT_EYE_TOP, LEFT_EYE_BOTTOM, w, h)
        right_corners = self._eye_corners(landmarks,
                                            RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER,
                                            RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, w, h)

        left_h  = self._horizontal_ratio(left_ic,  left_corners)
        right_h = self._horizontal_ratio(right_ic, right_corners)
        avg_h   = (left_h + right_h) / 2.0

        left_v  = self._vertical_ratio(left_ic,  left_corners)
        right_v = self._vertical_ratio(right_ic, right_corners)
        avg_v   = (left_v + right_v) / 2.0

        # ── Blink (Eye Aspect Ratio) ───────────────────────────────────────
        left_ear  = self._eye_aspect_ratio(landmarks, LEFT_EYE_INDICES,  w, h)
        right_ear = self._eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0
        blink     = self._detect_blink(avg_ear)

        # ── Classify & update ─────────────────────────────────────────────
        raw_direction = self._classify_gaze(avg_h, avg_v, blink)
        direction     = self._smooth_direction(raw_direction)
        self._update_metrics(direction, blink)
        attn_score = self._compute_attention_score()

        # ── Build GazeFrame ───────────────────────────────────────────────
        gf = GazeFrame(
            timestamp=time.time(),
            gaze_direction=direction,
            left_ratio=left_h,
            right_ratio=right_h,
            vertical_ratio=avg_v,
            blink_detected=blink,
            attention_score=attn_score,
            ear=avg_ear,
        )
        self.session_data.append(gf)

        # ── Draw overlays ─────────────────────────────────────────────────
        self._draw_overlays(annotated, landmarks, w, h,
                            left_ic, right_ic,
                            direction, attn_score, blink, avg_ear)

        return annotated, gf

    def get_metrics(self) -> AttentionMetrics:
        return self.metrics

    def save_session_csv(self, filepath: str = "gaze_session.csv"):
        """Export all frame data to a CSV file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "gaze_direction", "left_ratio", "right_ratio",
                "vertical_ratio", "blink_detected", "attention_score", "ear"
            ])
            writer.writeheader()
            for gf in self.session_data:
                writer.writerow({
                    "timestamp":       round(gf.timestamp, 4),
                    "gaze_direction":  gf.gaze_direction,
                    "left_ratio":      round(gf.left_ratio, 4),
                    "right_ratio":     round(gf.right_ratio, 4),
                    "vertical_ratio":  round(gf.vertical_ratio, 4),
                    "blink_detected":  int(gf.blink_detected),
                    "attention_score": round(gf.attention_score, 2),
                    "ear":             round(gf.ear, 4),
                })
        print(f"[GazeTracker] Session CSV saved → {filepath}")

    def reset_session(self):
        """Clear all session data and restart metrics."""
        self.session_data.clear()
        self.gaze_history.clear()
        self.dir_history.clear()
        self.fixation_history.clear()
        self.metrics          = AttentionMetrics()
        self._prev_direction  = "unknown"
        self._in_fixation     = False

    # ─────────────────────────────────────────────────────────
    #  Geometry helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _iris_center(landmarks, indices: List[int], w: int, h: int) -> Tuple[int, int]:
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        cx, cy = pts.mean(axis=0).astype(int)
        return int(cx), int(cy)

    @staticmethod
    def _eye_corners(landmarks, left_idx, right_idx, top_idx, bottom_idx,
                            w: int, h: int) -> dict:
        def pt(idx):
            return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
        return {"left": pt(left_idx), "right": pt(right_idx),
                "top":  pt(top_idx),  "bottom": pt(bottom_idx)}

    @staticmethod
    def _horizontal_ratio(iris_center: Tuple[int, int], corners: dict) -> float:
        """
        Horizontal gaze ratio:  (iris_x − eye_left) / eye_width
        0.0 = far left edge, 0.5 = centre, 1.0 = far right edge.
        """
        eye_width = corners["right"][0] - corners["left"][0]
        if eye_width == 0:
            return 0.5
        ratio = (iris_center[0] - corners["left"][0]) / eye_width
        return float(np.clip(ratio, 0.0, 1.0))

    @staticmethod
    def _vertical_ratio(iris_center: Tuple[int, int], corners: dict) -> float:
        """
        Vertical gaze ratio:  (iris_y − eye_top) / eye_height
        0.0 = looking up, 0.5 = centre, 1.0 = looking down.
        """
        eye_height = corners["bottom"][1] - corners["top"][1]
        if eye_height == 0:
            return 0.5
        ratio = (iris_center[1] - corners["top"][1]) / eye_height
        return float(np.clip(ratio, 0.0, 1.0))

    @staticmethod
    def _eye_aspect_ratio(landmarks, indices: List[int], w: int, h: int) -> float:
        """
        Eye Aspect Ratio (EAR):
            EAR = (v1 + v2) / (2 × horizontal_distance)
        Falls toward 0 when the eye closes → used for blink detection.
        """
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        horiz = np.linalg.norm(pts[0] - pts[8])
        v1    = np.linalg.norm(pts[4] - pts[12])
        v2    = np.linalg.norm(pts[2] - pts[10])
        ear   = (v1 + v2) / (2.0 * horiz) if horiz > 0 else 0.3
        return float(ear)

    # ─────────────────────────────────────────────────────────
    #  Gaze classification & blink
    # ─────────────────────────────────────────────────────────

    def _classify_gaze(self, h_ratio: float, v_ratio: float, blink: bool) -> str:
        """Map normalised iris ratios to a gaze direction label."""
        if blink:
            return "blink"
        if v_ratio < self.VERTICAL_UP_THRESHOLD:
            return "up"
        if v_ratio > self.VERTICAL_DOWN_THRESHOLD:
            return "down"
        if h_ratio < self.HORIZONTAL_LEFT_THRESHOLD:
            return "left"
        if h_ratio > self.HORIZONTAL_RIGHT_THRESHOLD:
            return "right"
        return "center"

    def _smooth_direction(self, raw: str) -> str:
        """
        بيمنع الـ saccade count من إنه ينفجر بسبب تذبذب الـ iris.
        الاتجاه الجديد لازم يتكرر 3 فريمات متتاليين عشان يتعدّ تغيير حقيقي.
        blink بيعدّي فوراً من غير smoothing.
        """
        if raw == "blink":
            return raw
        self._dir_buffer.append(raw)
        # لو الـ buffer كله نفس الاتجاه → اعتمده
        if len(self._dir_buffer) == 3 and len(set(self._dir_buffer)) == 1:
            self._stable_direction = raw
        return self._stable_direction

    def _detect_blink(self, ear: float) -> bool:
        """
        يعدّ الرمشة بس لو EAR فضل تحت الـ threshold أكتر من فريمين متتاليين.
        ده بيمنع الـ noise من إنه يتعدّ رمشة.
        """
        now = time.time()

        # ── DEBUG: اطبع EAR كل نص ثانية عشان نعرف القيم الحقيقية ──────────
        if not hasattr(self, '_debug_timer'):
            self._debug_timer = 0.0
            self._ear_min = 9.0
        self._ear_min = min(self._ear_min, ear)
        if now - self._debug_timer >= 0.5:
            self._debug_timer = now
            print(f"[BLINK DEBUG] EAR={ear:.4f}  min_seen={self._ear_min:.4f}"
                  f"  threshold={self.EAR_BLINK_THRESHOLD}  frames_below={self._blink_frame_count}"
                  f"  total_blinks={self.metrics.blink_count}")
        # ────────────────────────────────────────────────────────────────────

        if ear < self.EAR_BLINK_THRESHOLD:
            self._blink_frame_count += 1
            if self._blink_frame_count >= self.EAR_BLINK_FRAMES and now > self._blink_cooldown:
                self._blink_cooldown    = now + self.BLINK_COOLDOWN_S
                self._blink_frame_count = 0
                print(f"[BLINK DEBUG] ✅ BLINK COUNTED!  total={self.metrics.blink_count + 1}")
                return True
        else:
            self._blink_frame_count = 0
        return False


    def _update_metrics(self, direction: str, blink: bool):
        m   = self.metrics
        now = time.time()

        m.total_frames += 1

        if blink:
            m.blink_count += 1

        if direction in m.direction_counts:
            m.direction_counts[direction] += 1

        active = direction not in ("blink", "unknown")

        # ── Fixation / Saccade tracking ───────────────────────────────────
        if active:
            if direction == self._current_fixation_dir:
                if not self._in_fixation:
                    self._in_fixation    = True
                    self._fixation_start = now
            else:
                # اتغير الاتجاه → اتحسب الـ fixation السابق لو كان كافي
                if self._in_fixation:
                    duration = now - self._fixation_start
                    if duration >= self.MIN_FIXATION_DURATION_S:
                        m.fixation_count         += 1
                        m.fixation_history_buf    = getattr(m, 'fixation_history_buf', [])
                        self.fixation_history.append(min(duration / 2.0, 1.0))
                    self._in_fixation = False

                if self._prev_direction not in ("blink", "unknown"):
                    m.saccade_count += 1

                self._current_fixation_dir = direction
        else:
            # blink أو unknown → نوقف الـ fixation الحالي
            if self._in_fixation:
                duration = now - self._fixation_start
                if duration >= self.MIN_FIXATION_DURATION_S:
                    m.fixation_count += 1
                    self.fixation_history.append(min(duration / 2.0, 1.0))
                self._in_fixation          = False
                self._current_fixation_dir = "unknown"

        # ── Duration tracking (مرة واحدة بس — frame_dt) ──────────────────
        # كانت المشكلة الرئيسية: الكود القديم بيضيف للـ fixation_duration مرتين:
        # مرة من الـ fixation block فوق ومرة من frame_dt هنا
        frame_dt = 1.0 / 30.0
        if direction == "center":
            m.total_fixation_duration    += frame_dt
        elif direction not in ("blink", "unknown"):
            m.total_distraction_duration += frame_dt
        # blink → مش center ومش distraction، بنتجاهلها في الـ duration

        self.gaze_history.append(1 if direction == "center" else 0)
        self.dir_history.append(direction)
        self._prev_direction = direction


    def _compute_attention_score(self) -> float:
        """
        Attention Score [0-100] يتبع الـ attention% مباشرة مع penalty بسيطة.

        المنطق:
          base  = centered_frac × 100          (الأساس هو نسبة النظر للشاشة)
          penalty = saccade_penalty + entropy_penalty  (خصم بسيط للتشتت)
          score = base - penalty  (محدود بين 0 و 100)

        بالشكل ده: 85% attention → score قريب من 85
        """
        if len(self.gaze_history) < 5:
            return 50.0

        centered_frac = sum(self.gaze_history) / len(self.gaze_history)

        # الأساس مباشر من نسبة النظر للشاشة
        base = centered_frac * 100.0

        # لو مش باصص خالص → 0 فوراً
        if centered_frac < 0.05:
            return 0.0

        # Saccade penalty (max 15 نقطة خصم)
        window_dirs = list(self.dir_history)
        window_secs = len(window_dirs) / 30.0
        transitions = sum(
            1 for i in range(1, len(window_dirs))
            if window_dirs[i] != window_dirs[i-1]
            and window_dirs[i]   not in ("blink", "unknown")
            and window_dirs[i-1] not in ("blink", "unknown")
        )
        saccade_rate = transitions / max(window_secs, 1.0)
        saccade_penalty = min(15.0, saccade_rate * 2.0)

        # Entropy penalty (max 10 نقط خصم) — لو الاتجاهات متوزعة بالتساوي
        from collections import Counter
        active = [d for d in window_dirs if d not in ("blink", "unknown")]
        if len(active) > 1:
            counts = Counter(active)
            total  = len(active)
            probs  = np.array([v / total for v in counts.values()])
            max_e  = np.log2(min(len(counts), 5))
            raw_e  = -np.sum(probs * np.log2(probs + 1e-9))
            entropy_penalty = (raw_e / max_e if max_e > 0 else 0.0) * 10.0
        else:
            entropy_penalty = 0.0

        score = base - saccade_penalty - entropy_penalty
        return float(np.clip(score, 0.0, 100.0))

    # ─────────────────────────────────────────────────────────
    #  Drawing / Overlay
    # ─────────────────────────────────────────────────────────

    _DIRECTION_COLOR = {
        "center":  (0,  220, 100),
        "left":    (0,  165, 255),
        "right":   (0,  165, 255),
        "up":      (255, 200,  0),
        "down":    (255, 200,  0),
        "blink":   (200, 200, 200),
        "unknown": (100, 100, 100),
    }

    def _draw_overlays(self, frame, landmarks, w, h,
                       left_ic, right_ic, direction, attn_score, blink, ear):
        color = self._DIRECTION_COLOR.get(direction, (200, 200, 200))

        iris_r = max(4, int(w * 0.012))
        cv2.circle(frame, left_ic,  iris_r, (0, 255, 200), 2)
        cv2.circle(frame, right_ic, iris_r, (0, 255, 200), 2)
        cv2.circle(frame, left_ic,  2, (0, 255, 200), -1)
        cv2.circle(frame, right_ic, 2, (0, 255, 200), -1)

        cv2.rectangle(frame, (10, 10), (290, 165), (15, 15, 15), -1)
        cv2.rectangle(frame, (10, 10), (290, 165), color, 2)

        cv2.putText(frame, f"GAZE: {direction.upper()}",
                    (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.putText(frame, f"EAR:      {ear:.3f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, f"BLINKS:   {self.metrics.blink_count}",
                    (20, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, f"SACCADES: {self.metrics.saccade_count}",
                    (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, f"FIXATIONS:{self.metrics.fixation_count}",
                    (20, 149), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        bar_x, bar_y = 10, h - 55
        bar_w, bar_h = w - 20, 35
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (25, 25, 25), -1)
        fill      = int(bar_w * attn_score / 100.0)
        bar_color = self._score_color(attn_score)
        if fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        cv2.putText(frame, f"Attention Score: {attn_score:.0f} / 100",
                    (bar_x + 10, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self._draw_gaze_compass(frame, w - 95, 80, direction, color)

    def _draw_gaze_compass(self, frame, cx, cy, direction, color):
        r = 50
        cv2.circle(frame, (cx, cy), r, (25, 25, 25), -1)
        cv2.circle(frame, (cx, cy), r, (70, 70, 70), 1)
        arrows = {
            "left":  ((cx - r + 10, cy), (cx - r // 2, cy)),
            "right": ((cx + r - 10, cy), (cx + r // 2, cy)),
            "up":    ((cx, cy - r + 10), (cx, cy - r // 2)),
            "down":  ((cx, cy + r - 10), (cx, cy + r // 2)),
        }
        for d, (p1, p2) in arrows.items():
            c = color if d == direction else (55, 55, 55)
            cv2.arrowedLine(frame, p1, p2, c, 2, tipLength=0.4)
        dot_color = color if direction == "center" else (55, 55, 55)
        cv2.circle(frame, (cx, cy), 7, dot_color, -1)

    @staticmethod
    def _score_color(score: float) -> Tuple[int, int, int]:
        """BGR colour: green ≥ 70, yellow ≥ 40, red < 40."""
        if score >= 70:
            return (30, 210, 70)
        if score >= 40:
            return (0, 185, 255)
        return (40, 60, 220)

    @staticmethod
    def _draw_no_face_overlay(frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (260, 55), (15, 15, 15), -1)
        cv2.putText(frame, "No Face Detected",
                    (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 80, 200), 2)