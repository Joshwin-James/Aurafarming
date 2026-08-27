import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time
import os
import glob
import random
import threading
import math
import subprocess
import platform

def play_audio(file_path, async_play=True):
    system = platform.system()
    cmd = []
    kwargs = {}
    if system == "Darwin":
        cmd = ['afplay', file_path]
    elif system == "Windows":
        cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', file_path]
        if hasattr(subprocess, 'CREATE_NO_WINDOW'):
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    else:
        cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', file_path]
        
    try:
        if async_play:
            return subprocess.Popen(cmd, **kwargs)
        else:
            return subprocess.run(cmd, **kwargs)
    except Exception as e:
        print(f"Failed to play audio {file_path}: {e}")
        return None

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDING_DIR = os.path.join(PROJECT_DIR, "recording")
SIGMA_VIDEO = os.path.join(PROJECT_DIR, "sigma.mp4")
WAITING_WAV = os.path.join(PROJECT_DIR, "waiting.wav")
EDIT_OUTPUT = os.path.join(PROJECT_DIR, "sigma_edit_output.mp4")

# ============== CONFIG ==============
RECORD_DURATION_SEC = 15     # How long to auto-record the user
CLIP_CHUNK_SEC = 3           # Split recording into chunks of this length
EDIT_CLIP_DURATION = 3       # Each clip segment in the final edit
EDIT_USER_CLIPS = 8          # Number of random user segments
EDIT_SIGMA_CLIPS = 0         # Number of random sigma.mp4 segments
FADE_FRAMES = 8              # Number of frames for quick dip-to-black transitions


# ============== CINEMATIC GRADE PIPELINE ==============

def cinematic_grade(frame):
    # High-quality cinematic sharpen (Unsharp Mask)
    blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
    img = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
    
    # Fast contrast boost
    img = cv2.convertScaleAbs(img, alpha=1.25, beta=-31.875)
    # Fast desaturate
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 0.85, gray_bgr, 0.15, 0)
    # Fast teal/orange
    b, g, r = cv2.split(img)
    b = cv2.add(b, 8)
    mean_val = int(cv2.mean(img)[0] * 0.02)
    r = cv2.add(r, mean_val)
    return cv2.merge([b, g, r])

def cinematic_grade_warm(frame):
    # Warm / Moody / Filmic
    img = cv2.convertScaleAbs(frame, alpha=1.1, beta=-15)
    
    # Soft Highlight Glow (Bloom)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    highlights = cv2.bitwise_and(img, img, mask=mask)
    blur = cv2.GaussianBlur(highlights, (0, 0), 10.0)
    img = cv2.addWeighted(img, 1.0, blur, 0.6, 0)
    
    # Warm colors: push red/orange, pull blue
    b, g, r = cv2.split(img)
    b = cv2.subtract(b, 15)
    r = cv2.add(r, 20)
    img = cv2.merge([b, g, r])
    
    # Fast desaturate
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2_bgr = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 0.85, gray2_bgr, 0.15, 0)
    
    return img

_vignette_cache = {}

def apply_vignette(frame, strength=0.65):
    rows, cols = frame.shape[:2]
    shape_key = (rows, cols)
    if shape_key not in _vignette_cache:
        kx = cv2.getGaussianKernel(cols, cols * 0.5)
        ky = cv2.getGaussianKernel(rows, rows * 0.5)
        mask = (ky * kx.T)
        mask = mask / mask.max()
        mask = 1 - strength * (1 - mask)
        _vignette_cache[shape_key] = np.dstack([mask.astype(np.float32)]*3)
    
    noisy = frame.astype(np.float32) * _vignette_cache[shape_key]
    return noisy.astype(np.uint8)

_noise_cache = {}
_noise_idx = 0

def apply_grain(frame, amount=10):
    global _noise_idx
    shape_key = frame.shape
    if shape_key not in _noise_cache:
        frames = []
        for _ in range(10):
            frames.append(np.random.randint(-amount, amount, shape_key, dtype=np.int16))
        _noise_cache[shape_key] = frames
            
    noise = _noise_cache[shape_key][_noise_idx % 10]
    _noise_idx += 1
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def apply_chromatic_aberration(frame, shift=6):
    b, g, r = cv2.split(frame)
    rows, cols = r.shape
    r = cv2.warpAffine(r, np.float32([[1,0,shift],[0,1,0]]), (cols, rows))
    b = cv2.warpAffine(b, np.float32([[1,0,-shift],[0,1,0]]), (cols, rows))
    return cv2.merge([b, g, r])

def apply_letterbox(frame, ratio=0.12):
    h, w = frame.shape[:2]
    bar = int(h * ratio / 2)
    frame[:bar, :] = 0
    frame[h - bar:, :] = 0
    return frame

def apply_zoom(frame, zoom_factor):
    if zoom_factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    nw, nh = int(w / zoom_factor), int(h / zoom_factor)
    x1, y1 = (w - nw) // 2, (h - nh) // 2
    return cv2.resize(frame[y1:y1+nh, x1:x1+nw], (w, h))

def vertical_slide_wipe_transition(clip_a, clip_b, transition_frames=4):
    if len(clip_a) < transition_frames or len(clip_b) < transition_frames:
        return clip_a + clip_b
        
    result = list(clip_a[:-transition_frames])
    h, w = clip_a[0].shape[:2]
    
    for i in range(transition_frames):
        alpha = (i + 1) / transition_frames
        y_offset = int(h * (1.0 - alpha))
        
        frame_a = clip_a[-(transition_frames - i)]
        frame_b = clip_b[i]
        
        comp = frame_a.copy()
        if y_offset < h:
            comp[y_offset:h, :] = frame_b[0:(h - y_offset), :]
            
        result.append(comp)
        
    result.extend(clip_b[transition_frames:])
    return result

def apply_impact_shake(frame, opacity=0.5):
    k = 31
    kernel = np.zeros((k, k))
    kernel[int((k-1)/2), :] = np.ones(k) / k
    blurred = cv2.filter2D(frame, -1, kernel)
    
    rows, cols = frame.shape[:2]
    M = np.float32([[1, 0, random.randint(-15, 15)], [0, 1, random.randint(-15, 15)]])
    shaken = cv2.warpAffine(blurred, M, (cols, rows))
    
    return cv2.addWeighted(frame, 1.0 - opacity, shaken, opacity, 0)

def apply_phonk_wasted_stinger(clip):
    if len(clip) == 0: return clip
    res = []
    text = "WASTED"
    font = cv2.FONT_HERSHEY_TRIPLEX
    scale = 3.0
    thick = 6
    
    for i, frame in enumerate(clip):
        if i == 0:
            res.append(np.full_like(frame, 255))
        elif i < 12:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=-30)
            f = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            progress = min(1.0, (i - 1) / 5.0)
            b = int(255 * (1.0 - progress) + 50 * progress)
            g = int(255 * (1.0 - progress) + 50 * progress)
            r = int(255 * (1.0 - progress) + 255 * progress)
            
            ts = cv2.getTextSize(text, font, scale, thick)[0]
            tx = (f.shape[1] - ts[0]) // 2
            ty = 640 + ts[1] // 2
            
            cv2.putText(f, text, (tx, ty), font, scale, (0, 0, 0), thick + 5)
            cv2.putText(f, text, (tx, ty), font, scale, (b, g, r), thick)
            res.append(f)
        else:
            res.append(frame)
    return res

def apply_cutout_slide(clip):
    if len(clip) == 0: return clip
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    first_frame = clip[0]
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    h, w = first_frame.shape[:2]
    
    cutout = None
    mask = None
    
    if len(faces) > 0:
        (x, y, fw, fh) = max(faces, key=lambda f: f[2]*f[3])
        pad_x = int(fw * 0.8)
        pad_y_top = int(fh * 0.6)
        pad_y_bot = int(fh * 1.5)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y_top)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y_bot)
        
        cutout_w = x2 - x1
        cutout_h = y2 - y1
        
        if cutout_w > 0 and cutout_h > 0:
            cutout_img = first_frame[y1:y2, x1:x2].copy()
            mask = np.zeros((cutout_h, cutout_w), dtype=np.float32)
            center = (cutout_w // 2, cutout_h // 2)
            axes = (cutout_w // 2, cutout_h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            
            scale_factor = 1.6
            new_w = int(cutout_w * scale_factor)
            new_h = int(cutout_h * scale_factor)
            
            cutout = cv2.resize(cutout_img, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h))
            
    res = []
    fc = len(clip)
    
    for i, frame in enumerate(clip):
        out = frame.copy()
        if cutout is not None:
            progress = min(1.0, i / (fc * 0.4))
            ease = 1.0 - (1.0 - progress) ** 3 
            
            c_h, c_w = cutout.shape[:2]
            final_x = (w - c_w) // 2
            final_y = h - c_h + int(h * 0.1)
            start_y = h
            
            curr_x = final_x
            curr_y = int(start_y + (final_y - start_y) * ease)
            
            y1 = max(0, curr_y)
            y2 = min(h, curr_y + c_h)
            x1 = max(0, curr_x)
            x2 = min(w, curr_x + c_w)
            
            cy1 = 0 if curr_y >= 0 else -curr_y
            cy2 = cy1 + (y2 - y1)
            cx1 = 0 if curr_x >= 0 else -curr_x
            cx2 = cx1 + (x2 - x1)
            
            if y2 > y1 and x2 > x1:
                roi = out[y1:y2, x1:x2].astype(np.float32)
                m = mask[cy1:cy2, cx1:cx2]
                m_inv = 1.0 - m
                
                c_crop = cutout[cy1:cy2, cx1:cx2].astype(np.float32)
                for c in range(3):
                    roi[:, :, c] = roi[:, :, c] * m_inv + c_crop[:, :, c] * m
                    
                out[y1:y2, x1:x2] = roi.astype(np.uint8)
                
        res.append(out)
    return res


# ============== SIGMA EDIT GENERATOR ==============

def center_crop_and_resize(img, target_size):
    target_w, target_h = target_size
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    img_aspect = w / h
    
    if img_aspect > target_aspect:
        # Image is wider than target aspect. Crop width.
        new_w = int(h * target_aspect)
        x_start = (w - new_w) // 2
        img_cropped = img[:, x_start:x_start+new_w]
    else:
        # Image is taller than target aspect. Crop height.
        new_h = int(w / target_aspect)
        y_start = (h - new_h) // 2
        img_cropped = img[y_start:y_start+new_h, :]
        
    return cv2.resize(img_cropped, target_size)

def face_crop_and_resize(img, target_size, face_bbox):
    target_w, target_h = target_size
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    img_aspect = w / h
    
    if face_bbox is None:
        cx, cy = w // 2, h // 2
    else:
        fx, fy, fw, fh = face_bbox
        cx = fx + fw // 2
        cy = fy + fh // 2
        
    if img_aspect > target_aspect:
        new_w = int(h * target_aspect)
        x_start = cx - new_w // 2
        if x_start < 0: x_start = 0
        if x_start + new_w > w: x_start = w - new_w
        img_cropped = img[:, x_start:x_start+new_w]
    else:
        new_h = int(w / target_aspect)
        y_start = cy - new_h // 2
        if y_start < 0: y_start = 0
        if y_start + new_h > h: y_start = h - new_h
        img_cropped = img[y_start:y_start+new_h, :]
        
    return cv2.resize(img_cropped, target_size)

def extract_clips(video_files, clip_dur, num_clips, target_size, fps=30):
    if not video_files:
        return []
    clips = []
    for _ in range(num_clips):
        path = random.choice(video_files)
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vfps = cap.get(cv2.CAP_PROP_FPS) or fps
        needed = int(clip_dur * vfps)
        start = random.randint(0, max(0, total - needed))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(needed):
            ret, f = cap.read()
            if not ret:
                break
            frames.append(center_crop_and_resize(f, target_size))
        cap.release()
        if frames:
            clips.append(frames)
    return clips

def dip_to_black_transition(clip_a, clip_b, fade_frames=FADE_FRAMES):
    if len(clip_a) < fade_frames or len(clip_b) < fade_frames:
        return clip_a + clip_b
    
    result = list(clip_a[:-fade_frames])
    
    # Fade out A to black
    for i in range(fade_frames):
        alpha = 1.0 - (i / fade_frames)
        faded = cv2.convertScaleAbs(clip_a[-(fade_frames - i)], alpha=alpha, beta=0)
        result.append(faded)
        
    # Fade in B from black
    for i in range(fade_frames):
        alpha = i / fade_frames
        faded = cv2.convertScaleAbs(clip_b[i], alpha=alpha, beta=0)
        result.append(faded)
        
    result.extend(clip_b[fade_frames:])
    return result

def zoom_flash_transition(clip_a, clip_b, transition_frames=6):
    if len(clip_a) < transition_frames or len(clip_b) < transition_frames:
        return clip_a + clip_b
        
    result = list(clip_a[:-transition_frames])
    
    for i in range(transition_frames):
        alpha = (i + 1) / transition_frames
        zoom_factor = 1.0 + alpha * 0.3
        zoomed = apply_zoom(clip_a[-(transition_frames - i)], zoom_factor)
        white = np.full(zoomed.shape, 255, dtype=np.uint8)
        flashed = cv2.addWeighted(zoomed, 1.0 - alpha, white, alpha, 0)
        result.append(flashed)
        
    for i in range(transition_frames):
        alpha = 1.0 - (i / transition_frames)
        zoom_factor = 1.0 + alpha * 0.3
        zoomed = apply_zoom(clip_b[i], zoom_factor)
        white = np.full(zoomed.shape, 255, dtype=np.uint8)
        flashed = cv2.addWeighted(zoomed, 1.0 - alpha, white, alpha, 0)
        result.append(flashed)
        
    result.extend(clip_b[transition_frames:])
    return result

def apply_speed_ramp(frames):
    n = len(frames)
    if n < 30: return frames
    out_frames = []
    p0 = (0, 0)
    p1 = (int(n * 0.15), int(n * 0.4))
    p2 = (int(n * 0.85), int(n * 0.6))
    p3 = (n-1, n-1)
    
    def get_j(i):
        if i <= p1[0]:
            progress = i / p1[0] if p1[0] > 0 else 0
            return p0[1] + progress * (p1[1] - p0[1])
        elif i <= p2[0]:
            progress = (i - p1[0]) / (p2[0] - p1[0])
            return p1[1] + progress * (p2[1] - p1[1])
        else:
            progress = (i - p2[0]) / (p3[0] - p2[0])
            return p2[1] + progress * (p3[1] - p2[1])
            
    for i in range(n):
        j_float = get_j(i)
        j_int = max(0, min(n-1, int(round(j_float))))
        out_frames.append(frames[j_int])
    return out_frames

def generate_edit(recording_dir, sigma_path, output_path, target_size, fps=30, edit_style="sigma", progress_cb=None):
    def p(msg):
        if progress_cb:
            progress_cb(msg)

    user_files = sorted(glob.glob(os.path.join(recording_dir, "clip_*.mp4")))
    
    if edit_style == "sigma":
        p("Extracting user clips...")
        user_clips = extract_clips(user_files, EDIT_CLIP_DURATION, EDIT_USER_CLIPS, target_size, fps)
        
        sigma_clips = []
        if EDIT_SIGMA_CLIPS > 0:
            p("Extracting sigma clips...")
            sigma_clips = extract_clips([sigma_path], EDIT_CLIP_DURATION, EDIT_SIGMA_CLIPS, target_size, fps)

        if not user_clips and not sigma_clips:
            p("No clips generated!")
            return False

        merged = []
        for i in range(max(len(user_clips), len(sigma_clips))):
            if i < len(user_clips):
                merged.append(apply_speed_ramp(user_clips[i]))
            if i < len(sigma_clips):
                merged.append(sigma_clips[i])

        p("Applying cinematic effects...")
        for ci, clip in enumerate(merged):
            for fi in range(len(clip)):
                if ci % 2 == 0:
                    clip[fi] = apply_zoom(clip[fi], 1.15)
                f = cinematic_grade(clip[fi])
                f = apply_grain(f)
                f = apply_vignette(f)
                clip[fi] = f

        p("Applying fade transitions...")
        final = merged[0]
        for i in range(min(FADE_FRAMES, len(final))):
            final[i] = cv2.convertScaleAbs(final[i], alpha=(i/FADE_FRAMES), beta=0)
            
        for i in range(1, len(merged)):
            final = dip_to_black_transition(final, merged[i])
            
        for i in range(min(FADE_FRAMES, len(final))):
            final[-(i+1)] = cv2.convertScaleAbs(final[-(i+1)], alpha=(i/FADE_FRAMES), beta=0)
            
    elif edit_style == "second":
        p("Extracting fast-cut clips for 'Second' edit...")
        # 1 Intro clip (3s) + 6 fast cuts (0.6s each)
        intro = extract_clips(user_files, 3.0, 1, target_size, fps)
        fast_cuts = extract_clips(user_files, 0.6, 6, target_size, fps)
        
        if not intro or not fast_cuts:
            p("No clips generated!")
            return False
            
        merged = intro + fast_cuts
        
        p("Applying warm filmic grade and Ken Burns zoom...")
        for ci, clip in enumerate(merged):
            frames_count = len(clip)
            for fi in range(frames_count):
                zoom_factor = 1.0 + (fi / frames_count) * 0.1 # Subtle zoom in
                f = apply_zoom(clip[fi], zoom_factor)
                f = cinematic_grade_warm(f)
                f = apply_grain(f, amount=5) # Gentle grain
                clip[fi] = f
                
        p("Applying zoom-flash transitions...")
        final = merged[0]
        for i in range(1, len(merged)):
            final = zoom_flash_transition(final, merged[i], transition_frames=6)
            
    elif edit_style == "third":
        p("Extracting 10s of continuous footage for 'Phonk Zoom' edit...")
        frames = []
        for f in user_files:
            cap = cv2.VideoCapture(f)
            while True:
                ret, frame = cap.read()
                if not ret or len(frames) >= 300:
                    break
                frames.append(center_crop_and_resize(frame, target_size))
            cap.release()
            if len(frames) >= 300:
                break
                
        if len(frames) < 300:
            while len(frames) < 300 and len(frames) > 0:
                frames.append(frames[-1].copy())
                
        if not frames:
            p("No clips generated!")
            return False
            
        p("Applying Phonk Zoom & Segments...")
        
        intro = frames[0:90]
        cuts = []
        for j in range(14):
            idx = 90 + j * 15
            cuts.append(frames[idx:idx+15])
            
        def process_zoom_seg(seg):
            res = []
            fc = len(seg)
            for fi in range(fc):
                zf = 1.0 + (fi / max(1, fc)) * 0.08
                res.append(apply_zoom(seg[fi], zf))
            return res
            
        segments = [process_zoom_seg(intro)]
        
        stinger_indices = random.sample(range(14), 2)
        cutout_indices = random.sample([i for i in range(14) if i not in stinger_indices], 2)
        
        for i, cut in enumerate(cuts):
            seg = process_zoom_seg(cut)
            if i in cutout_indices:
                seg = apply_cutout_slide(seg)
            if i in stinger_indices:
                seg = apply_phonk_wasted_stinger(seg)
            segments.append(seg)
            
        p("Applying Slide Wipe Transitions...")
        final = segments[0]
        for i in range(1, len(segments)):
            L1 = len(final)
            final = vertical_slide_wipe_transition(final, segments[i], transition_frames=4)
            if L1 < len(final): final[L1] = apply_impact_shake(final[L1])
            if L1+1 < len(final): final[L1+1] = apply_impact_shake(final[L1+1])

    p(f"Writing {len(final)} frames...")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, target_size)
    for f in final:
        writer.write(f)
    writer.release()
    
    p("Muxing audio into final video...")
    if edit_style == "sigma":
        audio_src = sigma_path
    elif edit_style == "second":
        audio_src = os.path.join(os.path.dirname(sigma_path), "second.mp4")
    else:
        audio_src = os.path.join(os.path.dirname(sigma_path), "third.wav")
        
    temp_out = output_path.replace(".mp4", "_temp.mp4")
    cmd = [
        "ffmpeg", "-y", 
        "-i", output_path, 
        "-i", audio_src, 
        "-c:v", "copy", 
        "-c:a", "aac", 
        "-map", "0:v:0", 
        "-map", "1:a:0", 
        "-shortest", 
        temp_out
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp_out, output_path)
    except Exception as e:
        print(f"Audio muxing failed: {e}")

    p("DONE")
    return True


# ============== STATE MACHINE ==============
# The app runs through these phases automatically:
#   WAITING  →  RECORDING  →  GENERATING  →  PLAYBACK
#                                                 ↓
#                                          (loops back to RECORDING)

STATE_WAITING    = "WAITING"
STATE_RECORDING  = "RECORDING"
STATE_GENERATING = "GENERATING"
STATE_PLAYBACK   = "PLAYBACK"


class FaceTrackerApp:
    PUNCH_DURATION = 0.35
    PUNCH_ZOOM_MAX = 1.18

    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg='#1a1a1a')

        self.width, self.height = 1280, 720

        self.canvas = tk.Canvas(
            window, width=self.width, height=self.height,
            bg='#1a1a1a', highlightthickness=0
        )
        self.canvas.pack(padx=20, pady=20)

        # Status bar
        self.status_var = tk.StringVar(value="INITIALIZING SYSTEM...")
        self.status_bar = tk.Label(
            window, textvariable=self.status_var,
            bg='#1a1a1a', fg='#64ff64',
            font=('Courier', 12, 'bold'), anchor='w'
        )
        self.status_bar.pack(fill='x', padx=20)

        # Phase info
        self.phase_label = tk.Label(
            window,
            text="FULLY AUTOMATIC — No input required",
            bg='#1a1a1a', fg='#888888',
            font=('Courier', 10), anchor='center'
        )
        self.phase_label.pack(fill='x', padx=20, pady=(5, 10))

        # State
        self.state = STATE_WAITING
        self.punch_start_time = None
        self.start_time = time.time()

        # Recording state
        self.clip_writer = None
        self.clip_index = 0
        self.record_start_time = 0

        # Edit playback
        self.edit_cap = None
        self.playback_start_time = 0
        self.playback_cycle = 0

        self.last_face_bbox = None
        self.frame_count = 0
        
        self.current_pip_bw = 500.0
        self.current_pip_bh = 500.0
        self.current_tph = 300.0
        self.edit_style_cycle = ["sigma", "second", "third"]
        self.edit_style_idx = 0
        self.edit_style = self.edit_style_cycle[self.edit_style_idx]

        # Camera / face detector (set in backend thread)
        self.cap = None
        self.face_cascade = None
        
        self.running = True
        self.audio_process = None

        os.makedirs(RECORDING_DIR, exist_ok=True)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.bind('<space>', lambda e: setattr(self, 'punch_start_time', time.time()))

        # Start backend init
        threading.Thread(target=self._init_backend, daemon=True).start()
        self.update()

    # ===== BACKEND INIT =====
    def _init_backend(self):
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if w > 0 and h > 0:
            ar = h / w
            self.width = 1280
            self.height = int(self.width * ar)
            self.window.after(0, lambda: self.canvas.config(width=self.width, height=self.height))
        print(f"Resolution: {self.width}x{self.height}")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.start_time = time.time()
        self.cam_ready = True
        # Transition: WAITING → RECORDING (after a 1 second delay for effect)
        self.window.after(1000, self._auto_start_recording)
        print("Camera ready!")

    # ===== AUTO STATE TRANSITIONS =====

    def _auto_start_recording(self):
        """Automatically begin recording"""
        self.state = STATE_RECORDING
        self.record_start_time = time.time()
        self._start_new_clip()
        self.status_var.set(f"● REC — Auto-recording for {RECORD_DURATION_SEC}s...")
        self.phase_label.config(text=f"PHASE 2/4: RECORDING ({RECORD_DURATION_SEC}s)")
        print(f"Auto-recording started for {RECORD_DURATION_SEC}s")

    def _auto_stop_and_generate(self):
        """Stop recording and generate the edit"""
        self._stop_clip_writer()
        self.state = STATE_GENERATING
        self.status_var.set("GENERATING SIGMA EDIT...")
        self.status_var.set("GENERATING EDIT...")
        self.phase_label.config(text="PHASE 3/4: GENERATING EDIT")
        print("Recording done. Generating edit...")

        def run():
            def progress(msg):
                self.window.after(0, lambda m=msg: self.status_var.set(f"EDIT: {m}"))
            
            if self.state == STATE_GENERATING:
                progress("Starting generation...")
                
                waiting_proc = None
                try:
                    waiting_proc = play_audio(WAITING_WAV, async_play=True)
                except: pass
                
                success = generate_edit(
                    RECORDING_DIR, SIGMA_VIDEO, EDIT_OUTPUT,
                    target_size=(720, 1280),
                    fps=30, edit_style=self.edit_style, progress_cb=progress
                )
                
                if waiting_proc:
                    try: waiting_proc.terminate()
                    except: pass
                
                if success:
                    try:
                        play_audio(os.path.join(PROJECT_DIR, 'confim.wav'), async_play=False)
                    except:
                        pass
                        
                self.window.after(0, lambda: self._auto_start_playback(success))

        threading.Thread(target=run, daemon=True).start()

    def _auto_start_playback(self, success):
        """Start playing the generated edit"""
        if not success:
            self._auto_restart_cycle()
            return
            
        self.state = STATE_PLAYBACK
        self.edit_cap = cv2.VideoCapture(EDIT_OUTPUT)
        self.playback_start_time = time.time()
        self.last_edit_frame_time = 0
        
        if getattr(self, 'audio_process', None):
            try: self.audio_process.terminate()
            except: pass
            
        if self.edit_style == "sigma":
            audio_file = SIGMA_VIDEO
        elif self.edit_style == "second":
            audio_file = os.path.join(PROJECT_DIR, "second.mp4")
        else:
            audio_file = os.path.join(PROJECT_DIR, "third.wav")
            
        self.audio_process = play_audio(audio_file, async_play=True)
        
        self.playback_cycle += 1
        self.last_edit_frame = None
        self.status_var.set(f"▶ PLAYING {self.edit_style.upper()} EDIT (Cycle {self.playback_cycle})")
        self.phase_label.config(text="PHASE 4/4: PLAYBACK — Edit plays in PIP")
        print("Edit playing!")

    def _auto_restart_cycle(self):
        """After playback ends, loop back to recording for a new cycle"""
        if getattr(self, 'audio_process', None):
            try: self.audio_process.terminate()
            except: pass
            
        if self.edit_cap:
            self.edit_cap.release()
            self.edit_cap = None
            
        self.edit_style_idx = (self.edit_style_idx + 1) % 3
        self.edit_style = self.edit_style_cycle[self.edit_style_idx]
            
        # Clean old clips so next cycle gets fresh footage
        for f in glob.glob(os.path.join(RECORDING_DIR, "clip_*.mp4")):
            os.remove(f)
        self.clip_index = 0
        self._auto_start_recording()

    # ===== RECORDING HELPERS =====

    def _start_new_clip(self):
        self._stop_clip_writer()
        self.clip_index += 1
        path = os.path.join(RECORDING_DIR, f"clip_{self.clip_index:03d}.mp4")
        self.clip_writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 1280)
        )
        self._clip_start = time.time()
        print(f"  Clip: {path}")

    def _stop_clip_writer(self):
        if self.clip_writer:
            self.clip_writer.release()
            self.clip_writer = None

    # ===== DRAWING =====

    def draw_waiting_screen(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (50, 45, 40)
        cx, cy = self.width // 2, self.height // 2
        r = 100
        color = (150, 255, 150)
        pulse = (math.sin(time.time() * 4) + 1) / 2

        pts = []
        for deg in [30, 90, 150, 210, 270, 330]:
            pts.append([int(cx + r * math.cos(math.radians(deg))),
                        int(cy + r * math.sin(math.radians(deg)))])
        pts = np.array(pts, np.int32)
        cv2.polylines(frame, [pts], True, color, 2)
        for deg in [90, 210, 330]:
            cv2.line(frame, (cx, cy),
                     (int(cx + r * math.cos(math.radians(deg))),
                      int(cy + r * math.sin(math.radians(deg)))), color, 2)
        cv2.line(frame, (cx, cy - r - 40), (cx, cy + r + 40), color, 2)
        cv2.line(frame, (cx - r - 60, cy), (cx + r + 60, cy), color, 2)
        cv2.circle(frame, (cx, cy - r - 60 - int(10 * pulse)), 3, color, -1)

        text = "INITIALIZING . . ."
        font = cv2.FONT_HERSHEY_DUPLEX
        ts = cv2.getTextSize(text, font, 1.0, 2)[0]
        tx, ty = cx - ts[0] // 2, cy + ts[1] // 2
        cv2.rectangle(frame, (tx-10, ty-ts[1]-10), (tx+ts[0]+10, ty+10), (50,45,40), -1)
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(frame, text, (tx, ty), font, 1.0, (255,255,255), 2)
        return frame

    def draw_rotating_cube_pip(self, pw, ph):
        frame = np.zeros((ph, pw, 3), dtype=np.uint8)
        
        if self.state == STATE_GENERATING:
            speed = 5.0
            color = (0, 0, 255)
            thick = 3
            text = "GENERATING EDIT..."
        else:
            speed = 1.0
            color = (0, 255, 0)
            thick = 2
            text = "WAITING..."
            
        t = time.time() * speed
        rx, ry = t * 0.5, t * 0.8
        
        cx, cy = pw // 2, ph // 2
        size = min(pw, ph) * 0.3
        
        pts = np.array([
            [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1],
            [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1]
        ], dtype=float)
        
        cos_x, sin_x = math.cos(rx), math.sin(rx)
        cos_y, sin_y = math.cos(ry), math.sin(ry)
        
        projected = []
        for p in pts:
            y_new = p[1]*cos_x - p[2]*sin_x
            z_new = p[1]*sin_x + p[2]*cos_x
            x_new = p[0]*cos_y + z_new*sin_y
            
            projected.append((int(cx + x_new*size), int(cy + y_new*size)))
            
        edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        
        for e in edges:
            cv2.line(frame, projected[e[0]], projected[e[1]], color, thick)
            
        cv2.putText(frame, text, (10, ph - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thick)
        
        if self.state == STATE_GENERATING:
            glow = (math.sin(time.time() * 15) + 1) * 0.5
            frame = cv2.addWeighted(frame, 1.0, frame, glow, 0)
            
        return frame

    def get_pip_frame(self, pw, ph):
        if self.state == STATE_PLAYBACK and self.edit_cap is not None:
            now = time.time()
            if getattr(self, 'last_edit_frame_time', 0) == 0:
                self.last_edit_frame_time = now
                self.last_edit_frame = None
                
            while self.last_edit_frame is None or now >= self.last_edit_frame_time + (1.0 / 30.0):
                ret, f = self.edit_cap.read()
                if ret:
                    self.last_edit_frame = cv2.resize(f, (pw, ph))
                    if getattr(self, 'last_edit_frame_time', 0) == 0:
                        self.last_edit_frame_time = now
                    else:
                        self.last_edit_frame_time += (1.0 / 30.0)
                else:
                    # Edit finished playing → restart cycle
                    self.window.after(0, self._auto_restart_cycle)
                    return self.draw_rotating_cube_pip(pw, ph)
            
            if self.last_edit_frame is not None:
                if self.last_edit_frame.shape[:2] != (ph, pw):
                    return cv2.resize(self.last_edit_frame, (pw, ph))
                return self.last_edit_frame
                
        return self.draw_rotating_cube_pip(pw, ph)

    def draw_cube(self, img, bbox):
        x, y, w, h = bbox
        
        # Scale down by 20% to make the cube smaller
        cx, cy = x + w//2, y + h//2
        w = int(w * 0.8)
        h = int(h * 0.8)
        x = cx - w//2
        y = cy - h//2
        
        px, py = int(w*0.1), int(h*0.1)
        x -= px; y -= py; w += px*2; h += py*2
        pf = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.int32)
        ox, oy = int(w*0.2), int(h*0.15)
        pb = np.array([[x+ox,y-oy],[x+w+ox,y-oy],[x+w+ox,y+h-oy],[x+ox,y+h-oy]], np.int32)
        c = (100,255,100)
        for i in range(4):
            cv2.line(img, tuple(pf[i]), tuple(pb[i]), c, 1)
        cv2.polylines(img, [pf], True, c, 2)
        cv2.polylines(img, [pb], True, c, 1)
        cv2.putText(img, "SUBJECT - LOCKED", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

    def draw_hud(self, frame):
        h, w = frame.shape[:2]

        if self.state == STATE_WAITING:
            status = "SYSTEM STANDBY"
            color = (0, 255, 255)
        elif self.state == STATE_RECORDING:
            status = "REC"
            color = (0, 0, 255)
        elif self.state == STATE_PLAYBACK:
            status = "PLAYBACK"
            color = (0, 255, 0)
        else:
            status = "" # Hide for GENERATING
        
        if status:
            cv2.putText(frame, status, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Recording countdown bar
        if self.state == STATE_RECORDING:
            progress = min(1.0, (time.time() - self.record_start_time) / RECORD_DURATION_SEC)
            bar_w = int((w - 40) * progress)
            cv2.rectangle(frame, (20, 50), (20 + bar_w, 56), (0, 200, 255), -1)
            cv2.rectangle(frame, (20, 50), (w - 20, 56), (100, 255, 100), 1)
            remaining = max(0, RECORD_DURATION_SEC - (time.time() - self.record_start_time))
            cv2.putText(frame, f"REC {remaining:.0f}s", (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # ===== MAIN UPDATE LOOP =====

    def update(self):
        if self.state == STATE_WAITING or not self.cam_ready:
            frame = self.draw_waiting_screen()
        else:
            # STATE_RECORDING, STATE_GENERATING, or STATE_PLAYBACK → show live feed with overlays
            frame = self._get_live_frame()
            if frame is None:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2_im))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        if self.running:
            self.window.after(15, self.update)

    def _get_live_frame(self):
        """Capture, process, and overlay a single live frame"""
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.width, self.height))

        # Fast Face Detection: Run every 3rd frame on a 4x downscaled image for massive CPU savings
        self.frame_count += 1
        if self.frame_count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            faces = self.face_cascade.detectMultiScale(small_gray, 1.1, 5, minSize=(15, 15))
            if len(faces) > 0:
                fx_, fy_, fw_, fh_ = max(faces, key=lambda r: r[2]*r[3])
                self.last_face_bbox = (fx_*4, fy_*4, fw_*4, fh_*4)

        # Save raw frame BEFORE effects
        if self.state == STATE_RECORDING and self.clip_writer is not None:
            # Face track and crop specifically for the recording so the final edit perfectly frames the user
            rec_frame = face_crop_and_resize(frame, (720, 1280), self.last_face_bbox)
            self.clip_writer.write(rec_frame)
            # Rotate clip chunks
            if time.time() - self._clip_start >= CLIP_CHUNK_SEC:
                self._start_new_clip()
            # Auto-stop after RECORD_DURATION_SEC
            if time.time() - self.record_start_time >= RECORD_DURATION_SEC:
                self._auto_stop_and_generate()

        # Cinematic pipeline
        frame = cinematic_grade(frame)

        zoom = 1.0
        in_punch = False
        if self.punch_start_time:
            el = time.time() - self.punch_start_time
            if el < self.PUNCH_DURATION:
                prog = el / self.PUNCH_DURATION
                zoom = 1.0 + (self.PUNCH_ZOOM_MAX - 1.0) * (1 - (1 - prog)**3)
                in_punch = True
            else:
                self.punch_start_time = None

        frame = apply_zoom(frame, zoom)
        if in_punch:
            frame = apply_chromatic_aberration(frame, shift=8)
        frame = apply_vignette(frame)
        frame = apply_grain(frame)

        # Overlays
        h, w = frame.shape[:2]

        if self.last_face_bbox is not None:
            self.draw_cube(frame, self.last_face_bbox)

        self.draw_hud(frame)

        # PIP Animation & Size
        target_pip_bw = 720.0 if self.state == STATE_PLAYBACK else 300.0
        target_pip_bh = 1280.0 if self.state == STATE_PLAYBACK else 300.0
        target_tph = int(h * 0.8) if self.state == STATE_PLAYBACK else int(h * 0.3)
        
        self.current_pip_bw += (target_pip_bw - self.current_pip_bw) * 0.15
        self.current_pip_bh += (target_pip_bh - self.current_pip_bh) * 0.15
        self.current_tph += (target_tph - self.current_tph) * 0.15

        tph = int(self.current_tph)
        tpw = int(self.current_pip_bw * (tph / self.current_pip_bh))
        
        pip = self.get_pip_frame(tpw, tph)
        cv2.rectangle(pip, (0,0), (tpw-1, tph-1), (255,255,0), 2)

        pad_y, pad_x = 40, 20
        sy, ey = pad_y, pad_y + tph
        sx, ex = w - pad_x - tpw, w - pad_x
        if sy >= 0 and ey <= h and sx >= 0 and ex <= w:
            frame[sy:ey, sx:ex] = pip

        label = "EDIT" if self.state == STATE_PLAYBACK else "CAM 01"
        cv2.putText(frame, label, (w - pad_x - 60, pad_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        frame = apply_letterbox(frame)
        return frame

    def on_closing(self):
        self.running = False
        if getattr(self, 'audio_process', None):
            try: self.audio_process.terminate()
            except: pass
        if self.cap:
            self.cap.release()
        if self.edit_cap:
            self.edit_cap.release()
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = FaceTrackerApp(root, "MONITOR")
    root.mainloop()
