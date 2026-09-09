document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const webcamEl = document.getElementById('webcam');
    const cameraStatusEl = document.getElementById('camera-status');
    const faceLockStatusEl = document.getElementById('face-lock-status');
    const targetBoxEl = document.getElementById('target-box');
    const targetLabelEl = document.getElementById('target-label');
    const recOverlayEl = document.getElementById('rec-overlay');
    const recTimerEl = document.getElementById('rec-timer');
    const consoleBoxEl = document.getElementById('console-box');
    
    // Style & Action Controls
    const styleCards = document.querySelectorAll('.style-card');
    const btnRecord = document.getElementById('btn-record');
    const btnAutoLoop = document.getElementById('btn-auto-loop');
    
    // Modal & Player
    const outputModal = document.getElementById('output-modal');
    const outputPlayer = document.getElementById('output-player');
    const btnDownload = document.getElementById('btn-download');
    const btnAgain = document.getElementById('btn-again');
    const closeModal = document.getElementById('close-modal');

    // Steppers
    const steps = {
        standby: document.getElementById('step-standby'),
        recording: document.getElementById('step-recording'),
        processing: document.getElementById('step-processing'),
        playback: document.getElementById('step-playback')
    };

    // State Variables
    let selectedStyle = 'sigma';
    let mediaRecorder = null;
    let recordedChunks = [];
    let isAutoLoop = false;
    let isProcessing = false;
    let stream = null;

    // Logging helper
    function log(message, type = 'info') {
        const line = document.createElement('div');
        line.className = `log-line ${type}`;
        line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        consoleBoxEl.appendChild(line);
        consoleBoxEl.scrollTop = consoleBoxEl.scrollHeight;
    }

    // Step state helper
    function setPhase(phase) {
        Object.keys(steps).forEach(k => steps[k].classList.remove('active'));
        if (steps[phase]) {
            steps[phase].classList.active = true;
            steps[phase].classList.add('active');
        }
    }

    // Initialize Webcam Stream
    async function initCamera() {
        try {
            log('Requesting webcam access...', 'info');
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 720 },
                    height: { ideal: 1280 },
                    facingMode: "user"
                },
                audio: false
            });

            webcamEl.srcObject = stream;
            
            cameraStatusEl.innerHTML = `
                <span class="status-dot active"></span>
                <span class="status-text">CAMERA LOCKED</span>
            `;
            faceLockStatusEl.textContent = "TARGET LOCK: TRACKING";
            log("Webcam connected successfully.", "success");
            startHUDAnimation();
        } catch (err) {
            log(`Camera error: ${err.message}`, 'warn');
            cameraStatusEl.innerHTML = `
                <span class="status-dot warning"></span>
                <span class="status-text">CAMERA BLOCKED</span>
            `;
            alert("Please allow webcam access in your browser to use AuraFarming.");
        }
    }

    // Dynamic HUD Box Animation Simulation
    function startHUDAnimation() {
        let angle = 0;
        setInterval(() => {
            if (isProcessing) return;
            angle += 0.05;
            const offsetX = Math.sin(angle) * 15;
            const offsetY = Math.cos(angle * 0.7) * 10;
            targetBoxEl.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
        }, 50);
    }

    // Style Card Selection
    styleCards.forEach(card => {
        card.addEventListener('click', () => {
            styleCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            selectedStyle = card.dataset.style;
            log(`Switched to edit cycle: ${selectedStyle.toUpperCase()}`, 'info');
        });
    });

    // Record Micro Clip from Browser Webcam
    function recordClip(durationSec = 4) {
        return new Promise((resolve, reject) => {
            if (!stream) {
                return reject(new Error("No webcam stream available"));
            }

            recordedChunks = [];
            let options = { mimeType: 'video/webm;codecs=vp9' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: 'video/webm' };
            }

            try {
                mediaRecorder = new MediaRecorder(stream, options);
            } catch (e) {
                mediaRecorder = new MediaRecorder(stream);
            }

            mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    recordedChunks.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'video/webm' });
                resolve(blob);
            };

            // Start Recording
            mediaRecorder.start();
            setPhase('recording');
            recOverlayEl.classList.remove('hidden');
            log(`Recording ${durationSec}s micro-clip...`, 'info');

            let remaining = durationSec;
            recTimerEl.textContent = `00:0${remaining}`;
            
            const timer = setInterval(() => {
                remaining--;
                recTimerEl.textContent = `00:0${Math.max(0, remaining)}`;
                if (remaining <= 0) {
                    clearInterval(timer);
                    recOverlayEl.classList.add('hidden');
                    if (mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                    }
                }
            }, 1000);
        });
    }

    // Submit Recorded Clip to Server for Edit Generation
    async function generateEdit(clipBlob) {
        setPhase('processing');
        log(`Generating ${selectedStyle.toUpperCase()} edit cycle on server...`, 'info');

        const formData = new FormData();
        formData.append('video', clipBlob, `user_clip.${clipBlob.type.includes('mp4') ? 'mp4' : 'webm'}`);
        formData.append('style', selectedStyle);

        try {
            const resp = await fetch('/api/generate-edit', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const errData = await resp.json();
                throw new Error(errData.detail || 'Server processing error');
            }

            const data = await resp.json();
            log(`Edit generated successfully: ${data.filename}`, 'success');
            return data.video_url;
        } catch (err) {
            log(`Generation error: ${err.message}`, 'warn');
            throw err;
        }
    }

    // Execute Main Edit Cycle
    async function runCycle() {
        if (isProcessing) return;
        isProcessing = true;
        btnRecord.disabled = true;

        try {
            const clipBlob = await recordClip(4);
            const videoUrl = await generateEdit(clipBlob);

            setPhase('playback');
            showModal(videoUrl);
        } catch (err) {
            setPhase('standby');
            alert(`Error: ${err.message}`);
        } finally {
            isProcessing = false;
            btnRecord.disabled = false;
        }
    }

    // Modal Control
    function showModal(videoUrl) {
        outputPlayer.src = videoUrl;
        btnDownload.href = videoUrl;
        outputModal.classList.remove('hidden');
        outputPlayer.play();
    }

    closeModal.addEventListener('click', () => {
        outputModal.classList.add('hidden');
        outputPlayer.pause();
        setPhase('standby');
        if (isAutoLoop) {
            setTimeout(runCycle, 2000);
        }
    });

    btnAgain.addEventListener('click', () => {
        outputModal.classList.add('hidden');
        outputPlayer.pause();
        runCycle();
    });

    // Button Click Listeners
    btnRecord.addEventListener('click', () => {
        isAutoLoop = false;
        runCycle();
    });

    btnAutoLoop.addEventListener('click', () => {
        isAutoLoop = !isAutoLoop;
        if (isAutoLoop) {
            btnAutoLoop.innerHTML = `<span class="btn-icon">⏹️</span><span class="btn-text">STOP AUTO LOOP</span>`;
            btnAutoLoop.classList.add('btn-primary');
            log('Autonomous Auto-Loop enabled.', 'success');
            runCycle();
        } else {
            btnAutoLoop.innerHTML = `<span class="btn-icon">🔄</span><span class="btn-text">START AUTO LOOP</span>`;
            btnAutoLoop.classList.remove('btn-primary');
            log('Autonomous Auto-Loop stopped.', 'info');
        }
    });

    // Start Webcam on Load
    initCamera();
});
