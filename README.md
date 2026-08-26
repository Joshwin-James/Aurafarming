# AuraFarming Auto-Editor

AuraFarming is a fully automated, real-time Python application that acts as your personal AI videographer and cinematic editor. It continuously monitors your face using a webcam, automatically records short clips, and instantly generates aggressive, highly-styled Instagram Reels/TikTok-style edits synced to music.

## Features

### Real-Time Face Tracking & HUD
The application uses OpenCV's Haar Cascades to lock onto the subject. Once a face is detected, it draws a futuristic 3D rotating HUD cube around the face to indicate that the subject is "locked".

### Fully Autonomous Loop
The app runs in a 4-phase infinite loop without requiring any manual interaction:
1. **Standby**: The system boots up and waits for initialization.
2. **Recording**: Once ready, it automatically records five short 3-second micro-clips of the subject using the webcam.
3. **Generating**: It seamlessly compiles these 5 clips using advanced OpenCV array math to generate a heavily stylized montage. (Background waiting music plays during this phase).
4. **Playback**: A confirmation chime rings, and the final generated edit plays back inside a floating picture-in-picture (PiP) window. Once the playback finishes, the cycle restarts immediately!

### 3-Cycle Dynamic Engine
The engine rotates between three entirely distinct cinematic edit styles:

1. **Cycle 1: "Sigma" Edit**
   - A moody, desaturated cinematic grade.
   - Features complex speed ramps (slowing down the middle of the cut and speeding up the ends).
   - Intense motion-blur flash transitions between cuts.
   
2. **Cycle 2: "Second" Edit**
   - A warm, filmic color grade with continuous Ken Burns push-in zooms.
   - Cinematic letterboxing (black bars).
   - Super-fast 0.6s micro-cuts synced perfectly to a warm audio track.
   
3. **Cycle 3: "Phonk Zoom" Edit (10 Seconds)**
   - Highly aggressive phonk montage featuring 14 incredibly fast micro-cuts.
   - **Sliding Cutout Overlay**: Dynamically extracts the subject's face/torso from the video using a feathered alpha mask, scales it up, and animates it sliding aggressively into the frame.
   - **Vertical Slide Wipes & Impact Shakes**: Custom 4-frame vertical wipes followed immediately by a directional motion-blur camera shake to sell the impact.
   - **GTA "WASTED" Stinger**: Randomly drops the clip to high-contrast Black & White and overlays an animated "WASTED" text title that shifts from white to pure red.

### Native Audio Muxing
The engine natively multiplexes (`ffmpeg`) the appropriate audio soundtrack (`sigma.mp4`, `second.mp4`, or `third.wav`) directly into the final generated `sigma_edit_output.mp4` video file, ensuring perfect A/V sync.

## Dependencies

The entire engine was built to be blazingly fast by relying almost entirely on native OpenCV matrix operations, avoiding heavy external video processing libraries.

- `opencv-python` (`cv2`)
- `numpy`
- `ffmpeg` (for audio extraction and muxing)
- `afplay` (macOS native audio player)

## Setup & Usage

1. Ensure you have the required audio and video soundtracks (`sigma.mp4`, `second.mp4`, `third.mp4`, `waiting.wav`, `confim.wav`) in the project directory.
2. Run the main script:
   ```bash
   python3 face_monitor.py
   ```
3. Stand in front of the camera and let the autonomous engine do the rest!
