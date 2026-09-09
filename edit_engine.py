import cv2
import numpy as np
import random
import os
import glob
import subprocess
import shutil

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
            ty = int(f.shape[0] * 0.7)
            
            cv2.putText(f, text, (tx, ty), font, scale, (0, 0, 0), thick + 5)
            cv2.putText(f, text, (tx, ty), font, scale, (b, g, r), thick)
            res.append(f)
        else:
            res.append(frame)
    return res

def apply_cutout_slide(clip):
    if len(clip) == 0: return clip
    first_frame = clip[0]
    h, w = first_frame.shape[:2]
    
    cutout = None
    mask = None
    faces = []
    
    try:
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    except Exception as e:
        print(f"Face cascade detection failed: {e}")
    
    if len(faces) > 0:
        (x, y, fw, fh) = max(faces, key=lambda f: f[2]*f[3])
    else:
        # Fallback to center frame if cascade fails or no face found
        fw, fh = int(w * 0.4), int(h * 0.3)
        x, y = (w - fw) // 2, (h - fh) // 2
        
    if True: # Execute cutout with either detected face or fallback
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

def center_crop_and_resize(img, target_size):
    target_w, target_h = target_size
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    img_aspect = w / h
    
    if img_aspect > target_aspect:
        new_w = int(h * target_aspect)
        x_start = (w - new_w) // 2
        img_cropped = img[:, x_start:x_start+new_w]
    else:
        new_h = int(w / target_aspect)
        y_start = (h - new_h) // 2
        img_cropped = img[y_start:y_start+new_h, :]
        
    return cv2.resize(img_cropped, target_size)

def normalize_video_file(input_path):
    ffmpeg_cmd = shutil.which("ffmpeg")
    if not ffmpeg_cmd:
        return input_path
    
    norm_path = input_path + "_norm.mp4"
    cmd = [
        ffmpeg_cmd, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-r", "30",
        "-pix_fmt", "yuv420p",
        norm_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(norm_path) and os.path.getsize(norm_path) > 0:
            return norm_path
    except Exception as e:
        print(f"Video normalization error: {e}")
    return input_path

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
        
        if total > 0 and total > needed:
            start = random.randint(0, total - needed)
        else:
            start = 0
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        clip_frames = []
        count = 0
        while cap.isOpened() and count < (needed if needed > 0 else 90):
            ret, f = cap.read()
            if not ret or f is None:
                break
            clip_frames.append(center_crop_and_resize(f, target_size))
            count += 1
            
        cap.release()
        
        # If video was shorter than needed, loop it to fill the duration
        if clip_frames:
            while len(clip_frames) < needed:
                clip_frames.append(clip_frames[-1])
            clips.append(clip_frames)
            
    return clips

def dip_to_black_transition(clip_a, clip_b, fade_frames=8):
    if len(clip_a) < fade_frames or len(clip_b) < fade_frames:
        return clip_a + clip_b
    
    result = list(clip_a[:-fade_frames])
    
    for i in range(fade_frames):
        alpha = 1.0 - (i / fade_frames)
        faded = cv2.convertScaleAbs(clip_a[-(fade_frames - i)], alpha=alpha, beta=0)
        result.append(faded)
        
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

def generate_edit(user_files, edit_style, base_dir, output_path, target_size=(360, 640), fps=30):
    fade_frames = 8
    sigma_path = os.path.join(base_dir, "sigma.mp4")
    
    # Normalize input files with FFmpeg for 100% OpenCV frame decoding reliability
    norm_user_files = []
    for f in user_files:
        norm_user_files.append(normalize_video_file(f))
    user_files = norm_user_files
    
    if edit_style == "sigma":
        user_clips = extract_clips(user_files, 3.0, 6, target_size, fps)
        if not user_clips:
            user_clips = extract_clips(user_files, 1.5, 4, target_size, fps)
        if not user_clips:
            raise ValueError("Could not extract video clips from uploaded file.")
            
        merged = []
        for i in range(len(user_clips)):
            merged.append(apply_speed_ramp(user_clips[i]))
        user_clips = None # Free raw frames to prevent OOM
        
        for ci, clip in enumerate(merged):
            for fi in range(len(clip)):
                if ci % 2 == 0:
                    clip[fi] = apply_zoom(clip[fi], 1.15)
                f = cinematic_grade(clip[fi])
                f = apply_grain(f)
                f = apply_vignette(f)
                clip[fi] = f

        final = merged[0][:-fade_frames] if len(merged[0]) > fade_frames else merged[0]
        for i in range(1, len(merged)):
            final.extend(dip_to_black_transition(merged[i-1], merged[i], fade_frames))
            if i < len(merged) - 1:
                final.extend(merged[i][fade_frames:-fade_frames])
            else:
                final.extend(merged[i][fade_frames:])
                
        # Fade out end
        for i in range(min(fade_frames, len(final))):
            final[-(i+1)] = cv2.convertScaleAbs(final[-(i+1)], alpha=(i/fade_frames), beta=0)
            
        audio_src = sigma_path
        
    elif edit_style == "second":
        intro = extract_clips(user_files, 3.0, 1, target_size, fps)
        fast_cuts = extract_clips(user_files, 0.6, 6, target_size, fps)
        
        if not intro or not fast_cuts:
            intro = extract_clips(user_files, 1.5, 1, target_size, fps)
            fast_cuts = extract_clips(user_files, 0.5, 4, target_size, fps)

        if not intro or not fast_cuts:
            raise ValueError("Could not extract video clips for Second edit.")
            
        merged = intro + fast_cuts
        
        for ci, clip in enumerate(merged):
            frames_count = len(clip)
            for fi in range(frames_count):
                zoom_factor = 1.0 + (fi / max(1, frames_count)) * 0.1
                f = apply_zoom(clip[fi], zoom_factor)
                f = cinematic_grade_warm(f)
                f = apply_grain(f, amount=5)
                clip[fi] = f
                
        final = merged[0][:-fade_frames] if len(merged[0]) > fade_frames else merged[0]
        for i in range(1, len(merged)):
            final.extend(zoom_flash_transition(merged[i-1], merged[i], fade_frames))
            if i < len(merged) - 1:
                final.extend(merged[i][fade_frames:-fade_frames])
            else:
                final.extend(merged[i][fade_frames:])
        
        audio_src = os.path.join(base_dir, "second.mp4")
            
    else: # "third" (Phonk Zoom)
        frames = []
        for f in user_files:
            cap = cv2.VideoCapture(f)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None or len(frames) >= 300:
                    break
                frames.append(center_crop_and_resize(frame, target_size))
            cap.release()
            if len(frames) >= 300:
                break
                
        if not frames:
            raise ValueError("No video frames could be decoded from uploaded clip.")

        while len(frames) < 300:
            frames = frames + frames
        frames = frames[:300]
            
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
            
        final = segments[0]
        for i in range(1, len(segments)):
            L1 = len(final)
            final = vertical_slide_wipe_transition(final, segments[i], transition_frames=4)
            if L1 < len(final): final[L1] = apply_impact_shake(final[L1])
            if L1+1 < len(final): final[L1+1] = apply_impact_shake(final[L1+1])

    # Write raw video to temp file
    temp_raw = output_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(temp_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, target_size)
    for f in final:
        writer.write(f)
    writer.release()
    
    # Audio Muxing with FFmpeg
    if edit_style == "sigma":
        audio_src = sigma_path
    elif edit_style == "second":
        audio_src = os.path.join(base_dir, "second.mp4")
    else:
        audio_src = os.path.join(base_dir, "third.wav")
        
    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd and os.path.exists(audio_src):
        cmd = [
            ffmpeg_cmd, "-y", 
            "-i", temp_raw, 
            "-i", audio_src, 
            "-c:v", "libx264", 
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", 
            "-map", "0:v:0", 
            "-map", "1:a:0", 
            "-shortest", 
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(temp_raw):
                os.remove(temp_raw)
            return output_path
        except Exception as e:
            print(f"FFmpeg audio muxing failed: {e}")
            
    # Fallback to web-compatible h264 without audio if ffmpeg mux failed
    if ffmpeg_cmd:
        cmd = [
            ffmpeg_cmd, "-y", 
            "-i", temp_raw, 
            "-c:v", "libx264", 
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(temp_raw):
                os.remove(temp_raw)
            return output_path
        except Exception:
            pass
            
    if os.path.exists(temp_raw):
        shutil.move(temp_raw, output_path)
    return output_path
