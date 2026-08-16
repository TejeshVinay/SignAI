(function () {
    'use strict';

    // ===== STATE =====
    const state = {
        currentPage: 'home',
        cameraOn: false,
        detectedSigns: [],
        detectedEmotions: {},
        sessionData: JSON.parse(localStorage.getItem('signai_sessions') || '[]'),
        currentSessionSigns: [],
        currentSessionEmotions: [],
        practiceIndex: 0,
        charts: {},
        pollingInterval: null,
        lastSign: null,
        lastEmotion: null,
        userIsEditing: false,
        stream: null
    };

    // ===== SIGN DETECTION GLOBALS =====
    let currentSignRepeatCount = 0;
    const MAX_REPEAT = 2;
    let signModel = null;
    let mpHands = null;
    let handTrackingRAF = null;
    let lastVideoTime = -1;
    const signWindow = [];
    const SIGN_WINDOW_SIZE = 5;
    let lastProcessTime = 0;
    let handStillTimer = null;

    const SIGN_LABELS = [
        'A', 'B', 'Bye', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'ILoveYou', 'J', 'K', 'L',
        'M', 'Meet', 'N', 'No', 'O', 'P', 'Please', 'Q', 'R', 'S', 'T', 'Tell', 'Thankyou',
        'U', 'V', 'W', 'X', 'Y', 'Yes', 'Z', 'del', 'space'
    ];

    const WORD_LABELS = new Set([
        "Bye", "Hello", "ILoveYou", "Meet", "No", "Please", "Tell", "Thankyou", "Yes"
    ]);

    let lastAppendedSign = null;
    let lastAppendTime = 0;
    const REPEAT_COOLDOWN = 1000; // ms

    // ===== ASL DATA =====
    const ASL_HANDS = {
        A: '✊', B: '🖐️', C: '🤏', D: '☝️', E: '✊', F: '🤌',
        G: '👈', H: '🤞', I: '🤙', J: '🤙', K: '✌️', L: '🤟',
        M: '✊', N: '✊', O: '👌', P: '👇', Q: '👇', R: '🤞',
        S: '✊', T: '✊', U: '✌️', V: '✌️', W: '🤟', X: '☝️',
        Y: '🤙', Z: '☝️'
    };

    const EMOTIONS = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust', 'Neutral'];
    const EMOTION_EMOJIS = {
        Happy: '😊', Sad: '😢', Angry: '😠', Surprise: '😲',
        Fear: '😨', Disgust: '🤢', Neutral: '😐'
    };

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // ===== BROWSER SIGN DETECTION =====

    async function initBrowserSignDetection() {
        // 1 — Load TF.js graph model
        signModel = await tf.loadLayersModel('/static/models/sign_model_tfjs/model.json');
        console.log('Sign model loaded');

        // 2 â€” HandLandmarker exposed on window.mpVision via vision_bundle.mjs script tag
        const { HandLandmarker, FilesetResolver } = window.mpVision;

        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

        mpHands = await HandLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 1
        });

        console.log('MediaPipe HandLandmarker ready');
    }

    function normalizeHandLandmarks(landmarks) {
        let arr = new Float32Array(63);
        for (let i = 0; i < 21; i++) {
            arr[i * 3] = landmarks[i].x;
            arr[i * 3 + 1] = landmarks[i].y;
            arr[i * 3 + 2] = landmarks[i].z;
        }

        // Subtract wrist (landmark 0) — translation invariant
        const wx = arr[0], wy = arr[1], wz = arr[2];
        for (let i = 0; i < 21; i++) {
            arr[i * 3] -= wx;
            arr[i * 3 + 1] -= wy;
            arr[i * 3 + 2] -= wz;
        }

        // Scale by max norm — scale invariant
        let maxNorm = 0;
        for (let i = 0; i < 21; i++) {
            const norm = Math.sqrt(
                arr[i * 3] ** 2 + arr[i * 3 + 1] ** 2 + arr[i * 3 + 2] ** 2
            );
            if (norm > maxNorm) maxNorm = norm;
        }
        if (maxNorm > 0) {
            for (let i = 0; i < 63; i++) arr[i] /= maxNorm;
        }

        return arr;
    }
    function onHandResults(results) {
        if (!state.cameraOn) {
            console.log('BLOCKED: cameraOn is false');
            return;
        }

        if (!results.landmarks || results.landmarks.length === 0) {
            clearTimeout(handStillTimer);
            handStillTimer = null;
            signWindow.length = 0;
            currentSignRepeatCount = 0;
            $('#overlaySign').querySelector('span').textContent = '—';
            state.lastSign = null;
            return;
        }

        const landmarks = results.landmarks[0];
        const normalized = normalizeHandLandmarks(landmarks);

        let stableSign = null;

        tf.tidy(() => {
            const input = tf.tensor2d([normalized], [1, 63]);
            const predTensor = signModel.predict(input);
            const predArray = Array.from(predTensor.dataSync());

            const maxIdx = predArray.indexOf(Math.max(...predArray));
            const confidence = predArray[maxIdx];

            console.log('Predicted:', SIGN_LABELS[maxIdx], 'conf:', confidence.toFixed(2));

            if (confidence >= 0.75) {
                const sign = SIGN_LABELS[maxIdx];
                signWindow.push(maxIdx);
                if (signWindow.length > SIGN_WINDOW_SIZE) signWindow.shift();

                const freq = {};
                signWindow.forEach(s => freq[s] = (freq[s] || 0) + 1);
                const stableIdx = parseInt(
                    Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0]
                );
                stableSign = SIGN_LABELS[stableIdx];
                console.log('stableSign set to:', stableSign);
            }
        });

        console.log('After tidy, stableSign:', stableSign);

        if (stableSign) {
            $('#overlaySign').querySelector('span').textContent = stableSign;
            clearTimeout(handStillTimer);

            console.log('Setting timer for:', stableSign);
            handStillTimer = setTimeout(() => {
                console.log('TIMER FIRED for:', stableSign);
                console.log('cameraOn:', state.cameraOn);
                console.log('lastSign:', state.lastSign);

                if (!state.cameraOn) return;

                console.log('Calling updateDetectedSign...');
                updateDetectedSign(stableSign);
                maybe_append_sign_frontend(stableSign);
                state.lastSign = stableSign;
            }, 500);

        } else {
            clearTimeout(handStillTimer);
            handStillTimer = null;
            signWindow.length = 0;
            $('#overlaySign').querySelector('span').textContent = '—';
            state.lastSign = null;
        }
    }
    // function onHandResults(results) {
    //     if (!state.cameraOn) return;

    //     if (!results.landmarks || results.landmarks.length === 0) {
    //         $('#overlaySign').querySelector('span').textContent = '—';
    //         state.lastSign = null;
    //         return;
    //     }

    //     const landmarks = results.landmarks[0];
    //     const normalized = normalizeHandLandmarks(landmarks);

    //     // tf.tidy auto-disposes all intermediate tensors â€” no memory leaks
    //     tf.tidy(() => {
    //         const input = tf.tensor2d([normalized], [1, 63]);
    //         const predTensor = signModel.predict(input);
    //         const predArray = Array.from(predTensor.dataSync());

    //         const maxIdx = predArray.indexOf(Math.max(...predArray));
    //         const confidence = predArray[maxIdx];

    //         if (confidence >= 0.75) {
    //             const sign = SIGN_LABELS[maxIdx];
    //             $('#overlaySign').querySelector('span').textContent = sign;

    //             if (sign !== state.lastSign) {
    //                 updateDetectedSign(sign);
    //                 state.lastSign = sign;
    //             }
    //             maybe_append_sign_frontend(sign);
    //         } else {
    //             $('#overlaySign').querySelector('span').textContent = '—';
    //             state.lastSign = null;
    //         }
    //     });
    // }
    function startHandTracking() {
        lastVideoTime = -1;
        let lastProcessTime = 0;

        function detect() {
            if (!state.cameraOn) return;

            const video = document.getElementById('videoStream');
            const now = performance.now();

            if (video && video.readyState >= 2 && mpHands) {
                if (video.currentTime !== lastVideoTime && now - lastProcessTime >= 750) {
                    lastVideoTime = video.currentTime;
                    lastProcessTime = now;
                    const results = mpHands.detectForVideo(video, now);
                    onHandResults(results);
                }
            }

            handTrackingRAF = requestAnimationFrame(detect);
        }

        detect();
    }
    // function startHandTracking() {
    //     lastVideoTime = -1;

    //     function detect() {
    //         if (!state.cameraOn) return; // exits RAF loop when camera stops

    //         const video = document.getElementById('videoStream');

    //         if (video && video.readyState >= 2 && mpHands) {
    //             // Only process new frames — official Google recommended pattern
    //             if (video.currentTime !== lastVideoTime) {
    //                 lastVideoTime = video.currentTime;
    //                 const results = mpHands.detectForVideo(video, performance.now());
    //                 onHandResults(results);
    //             }
    //         }

    //         handTrackingRAF = requestAnimationFrame(detect);
    //     }

    //     detect();
    // }

    function stopHandTracking() {
        if (handTrackingRAF) {
            cancelAnimationFrame(handTrackingRAF);
            handTrackingRAF = null;
        }
        lastVideoTime = -1;
    }

    // ===== SENTENCE BUILDING (FRONTEND) =====

    function maybe_append_sign_frontend(sign) {
        if (!sign || sign === 'None' || sign === '—') return;

        const now = Date.now();

        // New sign → reset counter
        if (lastAppendedSign !== sign) {
            currentSignRepeatCount = 1;
            appendToSentenceFrontend(sign);
            lastAppendedSign = sign;
            lastAppendTime = now;
            return;
        }

        // Same sign → block if limit reached
        if (currentSignRepeatCount >= MAX_REPEAT) return;

        // Cooldown check
        if (now - lastAppendTime >= REPEAT_COOLDOWN) {
            currentSignRepeatCount++;
            appendToSentenceFrontend(sign);
            lastAppendTime = now;
        }
    }

    function appendToSentenceFrontend(label) {
        // if (state.userIsEditing) return;
        const builder = $('#sentenceBuilder');
        let current = builder.value;

        if (label === 'space') {
            builder.value = current + ' ';
        } else if (label === 'del') {
            builder.value = current.slice(0, -1);
        } else if (WORD_LABELS.has(label)) {
            if (current && !current.endsWith(' ')) current += ' ';
            builder.value = current + label + ' ';
        } else {
            builder.value = current + label;
        }

        updateWordChips();
    }

    // ===== NAVIGATION =====
    function initNavigation() {
        document.addEventListener('click', (e) => {
            const navEl = e.target.closest('[data-page]');
            if (navEl) {
                e.preventDefault();
                const page = navEl.getAttribute('data-page');
                navigateTo(page);
            }
        });
    }

    function navigateTo(page) {
        $$('.page').forEach(p => p.classList.remove('active'));
        const target = $(`#page-${page}`);
        if (target) target.classList.add('active');

        $$('.nav-link').forEach(link => {
            link.classList.toggle('active', link.getAttribute('data-page') === page);
        });

        $('#navLinks').classList.remove('open');
        $('#mobileToggle').classList.remove('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });

        state.currentPage = page;

        if (page === 'analytics') initCharts();
        if (page === 'learn') renderASLGrid();
        if (page === 'transcript') renderTranscripts();
    }

    // ===== THEME TOGGLE =====
    function initTheme() {
        const saved = localStorage.getItem('signai_theme') || 'dark';
        document.documentElement.setAttribute('data-theme', saved);
        updateThemeIcon(saved);

        $('#themeToggle').addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('signai_theme', next);
            updateThemeIcon(next);

            if (state.currentPage === 'analytics') {
                destroyCharts();
                initCharts();
            }
        });
    }

    function updateThemeIcon(theme) {
        const icon = $('#themeIcon');
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    // ===== MOBILE MENU =====
    function initMobileMenu() {
        $('#mobileToggle').addEventListener('click', () => {
            $('#navLinks').classList.toggle('open');
            $('#mobileToggle').classList.toggle('active');
        });
    }

    // ===== NAVBAR SCROLL =====
    function initNavScroll() {
        window.addEventListener('scroll', () => {
            $('#navbar').classList.toggle('scrolled', window.scrollY > 20);
        });
    }

    // ===== CAMERA / DETECTION =====
    function initDetection() {
        $('#btnOpenCamera').addEventListener('click', startCamera);
        $('#btnStopCamera').addEventListener('click', stopCamera);

        document.addEventListener('keydown', (e) => {
            if ((e.key === 'q' || e.key === 'Q' || e.key === 'Escape') && state.cameraOn) {
                stopCamera();
            }
        });

        $('#btnClearSentence').addEventListener('click', () => {
            $('#sentenceBuilder').value = '';
            $('#wordChips').innerHTML = '';
            state.currentSessionSigns = [];
            lastAppendedSign = null;
            lastAppendTime = 0;
        });

        $('#btnInterpret').addEventListener('click', generateInterpretation);
        $('#btnPlayAudio').addEventListener('click', playTTS);
        $('#btnStopAudio').addEventListener('click', stopTTS);

        populateVoices();
        if (window.speechSynthesis) {
            window.speechSynthesis.onvoiceschanged = populateVoices;
        }

        $('#sentenceBuilder').addEventListener('focus', () => { state.userIsEditing = true; });
        $('#sentenceBuilder').addEventListener('blur', () => { state.userIsEditing = false; });

        window.addEventListener('beforeunload', () => {
            if (state.cameraOn) navigator.sendBeacon('/stop_camera');
        });
    }

    async function startCamera() {
        // Step 1 — acquire webcam stream
        try {
            state.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' },
                audio: false
            });
        } catch (err) {
            showToast('Camera access denied. Please allow camera permission.', 'error');
            console.error('getUserMedia failed:', err);
            return;
        }

        state.cameraOn = true;

        // Step 2 — reset session data
        state.detectedSigns = [];
        state.detectedEmotions = {};
        state.currentSessionSigns = [];
        state.currentSessionEmotions = [];
        state.lastSign = null;
        state.lastEmotion = null;
        lastAppendedSign = null;
        lastAppendTime = 0;

        // Step 3 — reset UI panels
        $('#signPanel').innerHTML = '<div class="empty-state"><i class="fas fa-hand"></i><p>Signs will appear here during detection</p></div>';
        $('#emotionPanel').innerHTML = '<div class="empty-state"><i class="fas fa-face-smile"></i><p>Emotions will appear here during detection</p></div>';
        $('#signCount').textContent = '0';
        $('#emotionCount').textContent = '0';
        $('#overlaySign').querySelector('span').textContent = '—';
        $('#overlayEmotion').querySelector('span').textContent = '—';
        $('#sentenceBuilder').value = '';
        $('#wordChips').innerHTML = '';

        // Step 4 — show video feed immediately
        $('#cameraOffState').style.display = 'none';

        const video = document.getElementById('videoStream');
        if (!video) {
            showToast('Video element not found in HTML.', 'error');
            state.cameraOn = false;
            return;
        }
        video.srcObject = state.stream;
        video.style.display = 'block';
        await video.play();

        // Step 5 — update UI buttons and status
        $('#btnOpenCamera').style.display = 'none';
        $('#btnStopCamera').style.display = 'inline-flex';
        $('#controlHint').style.display = 'inline';
        $('#videoWrapper').classList.add('camera-on');
        $('.status-dot').classList.remove('offline');
        $('.status-dot').classList.add('online');
        $('.status-text').textContent = 'Live Detection Running';

        showToast('Camera started â€” loading sign model...', 'info');

        // Step 6 — start emotion polling immediately
        startFrameSending();

        // Step 7 — load sign model + MediaPipe in background (non-blocking)
        try {
            await initBrowserSignDetection();
            startHandTracking();
            showToast('Sign detection ready!', 'success');
        } catch (err) {
            console.error('Sign model init failed:', err);
            showToast('Sign model unavailable — check console for details', 'error');
        }
    }

    function stopCamera() {
        state.cameraOn = false;
        state.lastSign = null;
        state.lastEmotion = null;
        currentSignRepeatCount = 0;

        // Stop hand tracking RAF loop
        stopHandTracking();

        // Stop emotion polling interval
        if (state.pollingInterval) {
            clearInterval(state.pollingInterval);
            state.pollingInterval = null;
        }

        // Release webcam hardware
        if (state.stream) {
            state.stream.getTracks().forEach(track => track.stop());
            state.stream = null;
        }

        // Hide video element
        const video = document.getElementById('videoStream');
        if (video) {
            video.srcObject = null;
            video.style.display = 'none';
        }

        // Notify backend
        fetch("/stop_camera", { method: "POST" }).catch((err) => {
            console.warn('Could not reach /stop_camera:', err.message);
        });

        // Restore UI — keep sentence and panels visible
        $('#cameraOffState').style.display = 'flex';
        $('#btnOpenCamera').style.display = 'inline-flex';
        $('#btnStopCamera').style.display = 'none';
        $('#controlHint').style.display = 'none';
        $('#videoWrapper').classList.remove('camera-on');
        $('.status-dot').classList.remove('online');
        $('.status-dot').classList.add('offline');
        $('.status-text').textContent = 'Camera OFF';

        saveSession();
        state.detectedEmotions = {};
        showToast('Camera stopped — session saved', 'info');
    }

    // ===== EMOTION BACKEND POLLING =====
    function startFrameSending() {
        if (state.pollingInterval) clearInterval(state.pollingInterval);

        const canvas = document.getElementById('captureCanvas');
        const ctx = canvas.getContext('2d');
        let sending = false;

        state.pollingInterval = setInterval(async () => {
            if (!state.cameraOn || sending) return;

            const video = document.getElementById('videoStream');
            if (!video || video.readyState < 2) return;

            ctx.drawImage(video, 0, 0, 640, 480);
            const frameData = canvas.toDataURL('image/jpeg', 0.7);

            sending = true;
            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frame: frameData })
                });

                if (!res.ok) return;

                const data = await res.json();

                // Only emotion — sign is handled in browser

                if (data.emotion && data.emotion !== 'Unknown' && data.emotion !== state.lastEmotion) {
                    updateDetectedEmotion(data.emotion);
                }
                state.lastEmotion = data.emotion;

                $('#overlayEmotion').querySelector('span').textContent =
                    (data.emotion && data.emotion !== 'Unknown')
                        ? `${EMOTION_EMOJIS[data.emotion] || ''} ${data.emotion}`
                        : '—';

            } catch (err) {
                console.warn('Frame send failed:', err.message);
                $('#overlayEmotion').querySelector('span').textContent = '—';
            } finally {
                sending = false;
            }
        }, 500);
    }

    // ===== DETECTION PANEL UPDATES =====
    function updateDetectedSign(sign) {
        if (state.lastSign === sign && Date.now() - lastLogTime < 3000) return;
        console.log('updateDetectedSign called with:', sign);
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        const panel = $('#signPanel');
        console.log('panel found:', panel);

        const MAX_ITEMS = 50;
        if (panel.children.length > MAX_ITEMS) panel.removeChild(panel.lastChild);

        state.detectedSigns.push({ sign, time: timeStr });
        state.currentSessionSigns.push(sign);

        const empty = panel.querySelector('.empty-state');
        if (empty) empty.remove();

        const item = document.createElement('div');
        item.className = 'detection-item';
        item.innerHTML = `<span class="det-value">${sign}</span><span class="det-time">${timeStr}</span>`;
        panel.insertBefore(item, panel.firstChild);

        $('#signCount').textContent = state.detectedSigns.length;
        panel.scrollTop = 0;
    }

    function updateDetectedEmotion(emotion) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        const panel = $('#emotionPanel');

        const MAX_ITEMS = 50;
        if (panel.children.length > MAX_ITEMS) panel.removeChild(panel.lastChild);

        state.detectedEmotions[emotion] = (state.detectedEmotions[emotion] || 0) + 1;
        state.currentSessionEmotions.push(emotion);

        const empty = panel.querySelector('.empty-state');
        if (empty) empty.remove();

        const item = document.createElement('div');
        item.className = 'detection-item emotion-item';
        item.innerHTML = `
            <span class="det-value">${EMOTION_EMOJIS[emotion] || ''} ${emotion}</span>
            <span class="det-time">${timeStr}</span>
        `;
        panel.insertBefore(item, panel.firstChild);

        $('#emotionCount').textContent = state.currentSessionEmotions.length;
        panel.scrollTop = 0;
    }

    function updateWordChips() {
        const chips = $('#wordChips');
        chips.innerHTML = '';
        const text = $('#sentenceBuilder').value.trim();
        if (!text) return;

        const words = text.split('');
        words.forEach((char, i) => {
            if (char === ' ') return;
            const chip = document.createElement('span');
            chip.className = 'word-chip';
            chip.innerHTML = `${char} <i class="fas fa-times chip-remove"></i>`;
            chip.addEventListener('click', () => {
                const current = $('#sentenceBuilder').value;
                const arr = current.split('');
                arr.splice(i, 1);
                $('#sentenceBuilder').value = arr.join('');
                updateWordChips();
            });
            chips.appendChild(chip);
        });
    }

    // ===== AI INTERPRETATION =====
    async function generateInterpretation() {
        const sentence = $('#sentenceBuilder').value.trim();

        if (!sentence) {
            showToast('No signs detected yet. Start detection first.', 'error');
            return;
        }

        const responseDiv = $('#aiResponse');
        const responseText = $('#aiResponseText');
        const btn = $('#btnInterpret');

        responseDiv.style.display = 'block';
        responseText.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

        try {
  
            const gestures = sentence.split(" ");

            const res = await fetch("/generate_sentence", {
                method: "POST", // ✅ IMPORTANT
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    gestures: gestures,
                    emotion: state.lastEmotion || "neutral"
                })
            });

            if (!res.ok) throw new Error(`Server returned ${res.status}`);

            const data = await res.json();

            // ✅ Updated response handling
            const text = data.sentence || data.error || "No response";

            responseText.textContent = text;
            $('#ttsText').innerHTML = `<p style="color: var(--text-primary);">${text}</p>`;

            showToast('Interpretation generated successfully', 'success');

        } catch (err) {
            console.error('Interpretation failed:', err);
            responseText.textContent = 'Failed to generate interpretation.';
            showToast('Interpretation failed — check backend', 'error');
        }

        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-wand-magic-sparkles"></i> Generate Meaningful Interpretation';
    }

    // ===== TEXT TO SPEECH =====
    function populateVoices() {
        if (!window.speechSynthesis) return;
        const select = $('#voiceSelect');
        const voices = window.speechSynthesis.getVoices();

        select.innerHTML = '<option value="default">Default Voice</option>';
        voices.forEach((voice, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = `${voice.name} (${voice.lang})`;
            select.appendChild(opt);
        });
    }

    function playTTS() {
        if (!window.speechSynthesis) {
            showToast('Text-to-Speech not supported in this browser', 'error');
            return;
        }

        const text = $('#ttsText').textContent.trim();
        if (!text || text === 'Generated sentence will appear here for playback') {
            showToast('No text to speak. Generate an interpretation first.', 'error');
            return;
        }

        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        const voiceIdx = $('#voiceSelect').value;

        if (voiceIdx !== 'default') {
            const voices = window.speechSynthesis.getVoices();
            utterance.voice = voices[parseInt(voiceIdx)];
        }

        utterance.rate = 0.9;
        utterance.pitch = 1;
        window.speechSynthesis.speak(utterance);
        showToast('Playing audio...', 'info');
    }

    function stopTTS() {
        if (window.speechSynthesis) {
            window.speechSynthesis.cancel();
            showToast('Audio stopped', 'info');
        }
    }

    // ===== SESSION MANAGEMENT =====
    function saveSession() {
        if (state.currentSessionSigns.length === 0 && state.currentSessionEmotions.length === 0) return;

        const session = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            signs: [...state.currentSessionSigns],
            emotions: [...state.currentSessionEmotions],
            sentence: $('#sentenceBuilder').value,
            interpretation: $('#aiResponseText')?.textContent || ''
        };

        state.sessionData.push(session);
        localStorage.setItem('signai_sessions', JSON.stringify(state.sessionData));
        state.currentSessionSigns = [];
        state.currentSessionEmotions = [];
    }

    // ===== TRANSCRIPT PAGE =====
    function renderTranscripts() {
        const list = $('#transcriptList');
        const empty = $('#transcriptEmpty');
        list.querySelectorAll('.session-card').forEach(c => c.remove());

        if (state.sessionData.length === 0) {
            empty.style.display = 'flex';
            return;
        }

        empty.style.display = 'none';

        [...state.sessionData].reverse().forEach((session, i) => {
            const card = document.createElement('div');
            card.className = 'session-card';
            card.dataset.id = session.id;

            const date = new Date(session.timestamp);
            const dateStr = date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
            const timeStr = date.toLocaleTimeString();

            card.innerHTML = `
                <div class="session-card-header">
                    <div class="session-info">
                        <div class="session-icon"><i class="fas fa-clock-rotate-left"></i></div>
                        <div class="session-meta">
                            <h4>Session #${state.sessionData.length - i}</h4>
                            <span>${dateStr} at ${timeStr}</span>
                        </div>
                    </div>
                    <div class="session-actions">
                        <button class="btn btn-sm btn-ghost btn-save-session" title="Save corrections"><i class="fas fa-save"></i></button>
                        <button class="btn btn-sm btn-ghost btn-export-session" title="Export"><i class="fas fa-download"></i></button>
                        <button class="btn btn-sm btn-ghost btn-delete-session" title="Delete" style="color:var(--accent-danger);"><i class="fas fa-trash"></i></button>
                    </div>
                </div>
                <div class="session-card-body">
                    <textarea class="session-text">${session.sentence || session.signs.join('')}</textarea>
                    <div class="session-tags">
                        <span class="session-tag"><i class="fas fa-hand"></i> ${session.signs.length} signs</span>
                        <span class="session-tag"><i class="fas fa-face-smile"></i> ${session.emotions.length} emotions</span>
                        ${session.emotions.length > 0 ? `<span class="session-tag"><i class="fas fa-chart-simple"></i> Dominant: ${getMostFrequent(session.emotions)}</span>` : ''}
                    </div>
                </div>
            `;

            card.querySelector('.btn-delete-session').addEventListener('click', () => {
                state.sessionData = state.sessionData.filter(s => s.id !== session.id);
                localStorage.setItem('signai_sessions', JSON.stringify(state.sessionData));
                renderTranscripts();
                showToast('Session deleted', 'info');
            });

            card.querySelector('.btn-save-session').addEventListener('click', () => {
                const text = card.querySelector('.session-text').value;
                const idx = state.sessionData.findIndex(s => s.id === session.id);
                if (idx !== -1) {
                    state.sessionData[idx].sentence = text;
                    localStorage.setItem('signai_sessions', JSON.stringify(state.sessionData));
                    showToast('Session updated', 'success');
                }
            });

            card.querySelector('.btn-export-session').addEventListener('click', () => exportSession(session));
            list.appendChild(card);
        });
    }

    function exportSession(session) {
        const data = {
            session_id: session.id,
            date: new Date(session.timestamp).toLocaleString(),
            signs_detected: session.signs,
            emotions_detected: session.emotions,
            sentence: session.sentence,
            interpretation: session.interpretation
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `signai_session_${session.id}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Session exported', 'success');
    }

    function initTranscriptActions() {
        $('#btnExportAll').addEventListener('click', () => {
            if (state.sessionData.length === 0) { showToast('No sessions to export', 'error'); return; }
            const blob = new Blob([JSON.stringify(state.sessionData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `signai_all_sessions_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
            showToast('All sessions exported', 'success');
        });

        $('#btnClearHistory').addEventListener('click', () => {
            if (confirm('Are you sure you want to delete all session history?')) {
                state.sessionData = [];
                localStorage.removeItem('signai_sessions');
                renderTranscripts();
                showToast('History cleared', 'info');
            }
        });
    }

    // ===== LEARN ASL PAGE =====
    function renderASLGrid() {
        const grid = $('#aslGrid');
        grid.innerHTML = '';
        const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
        const search = $('#letterSearch').value.toUpperCase();
        const filtered = search ? letters.filter(l => l.includes(search)) : letters;

        filtered.forEach(letter => {
            const tile = document.createElement('div');
            tile.className = 'asl-tile';
            tile.innerHTML = `
                <span class="asl-tile-hand">${ASL_HANDS[letter]}</span>
                <span class="asl-tile-letter">${letter}</span>
                <span class="asl-tile-label">ASL Sign</span>
            `;
            tile.addEventListener('click', () => showPracticeModal(letters.indexOf(letter)));
            grid.appendChild(tile);
        });
    }

    function initLearnPage() {
        $('#letterSearch').addEventListener('input', renderASLGrid);
        $('#btnPracticeMode').addEventListener('click', () => { state.practiceIndex = 0; showPracticeModal(0); });
        $('#closePractice').addEventListener('click', () => { $('#practiceModal').style.display = 'none'; });
        $('#btnNextLetter').addEventListener('click', () => { state.practiceIndex = (state.practiceIndex + 1) % 26; updatePracticeModal(); });
        $('#btnPrevLetter').addEventListener('click', () => { state.practiceIndex = (state.practiceIndex - 1 + 26) % 26; updatePracticeModal(); });
        $('#practiceModal').addEventListener('click', (e) => { if (e.target === $('#practiceModal')) $('#practiceModal').style.display = 'none'; });
    }

    function showPracticeModal(index) {
        state.practiceIndex = index;
        updatePracticeModal();
        $('#practiceModal').style.display = 'flex';
    }

    function updatePracticeModal() {
        const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
        const letter = letters[state.practiceIndex];
        $('#practiceLetter').textContent = letter;
        $('#practiceHand').textContent = ASL_HANDS[letter];
    }

    // ===== ANALYTICS PAGE =====
    function destroyCharts() {
        Object.values(state.charts).forEach(chart => { if (chart) chart.destroy(); });
        state.charts = {};
    }

    function initCharts() {
        destroyCharts();
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const textColor = isDark ? '#a0a0b8' : '#555577';
        const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';

        $('#totalSessions').textContent = state.sessionData.length;
        const allSigns = state.sessionData.flatMap(s => s.signs);
        const allEmotions = state.sessionData.flatMap(s => s.emotions);
        $('#totalSigns').textContent = allSigns.length;
        $('#dominantEmotion').textContent = getMostFrequent(allEmotions) || '—';

        if (state.sessionData.length > 0) {
            const avgLen = Math.round(allSigns.length / state.sessionData.length);
            $('#avgDuration').textContent = `~${avgLen * 2}s`;
        }

        const sessionLabels = state.sessionData.map((s, i) => `S${i + 1}`);
        if (sessionLabels.length === 0) sessionLabels.push('S1', 'S2', 'S3', 'S4', 'S5');

        const emotionTimelineData = EMOTIONS.map(emo => {
            if (state.sessionData.length === 0) return sessionLabels.map(() => Math.floor(Math.random() * 5));
            return state.sessionData.map(s => s.emotions.filter(e => e === emo).length);
        });

        const emotionColors = ['#4ECB71', '#FF6B6B', '#FFB347', '#00D9FF', '#9B59B6', '#E67E22', '#95A5A6'];

        state.charts.timeline = new Chart($('#emotionTimeline').getContext('2d'), {
            type: 'line',
            data: {
                labels: sessionLabels,
                datasets: EMOTIONS.map((emo, i) => ({
                    label: emo, data: emotionTimelineData[i], borderColor: emotionColors[i],
                    backgroundColor: emotionColors[i] + '20', tension: 0.4, fill: false, pointRadius: 4, pointHoverRadius: 6
                }))
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { labels: { color: textColor, font: { family: 'Inter', size: 11 } } } },
                scales: {
                    x: { ticks: { color: textColor }, grid: { color: gridColor } },
                    y: { ticks: { color: textColor }, grid: { color: gridColor }, beginAtZero: true }
                }
            }
        });

        const emotionCounts = EMOTIONS.map(emo => {
            const count = allEmotions.filter(e => e === emo).length;
            return count || Math.floor(Math.random() * 10) + 1;
        });

        state.charts.distribution = new Chart($('#emotionDistribution').getContext('2d'), {
            type: 'doughnut',
            data: { labels: EMOTIONS, datasets: [{ data: emotionCounts, backgroundColor: emotionColors, borderWidth: 0, hoverOffset: 8 }] },
            options: {
                responsive: true, maintainAspectRatio: false, cutout: '65%',
                plugins: { legend: { position: 'right', labels: { color: textColor, font: { family: 'Inter', size: 11 }, padding: 12 } } }
            }
        });

        const signCounts = {};
        const signsToShow = allSigns.length > 0 ? allSigns : 'HELLOWORLD'.split('');
        signsToShow.forEach(s => { signCounts[s] = (signCounts[s] || 0) + 1; });
        const sortedSigns = Object.entries(signCounts).sort((a, b) => b[1] - a[1]).slice(0, 15);

        state.charts.frequency = new Chart($('#signFrequency').getContext('2d'), {
            type: 'bar',
            data: {
                labels: sortedSigns.map(s => s[0]),
                datasets: [{
                    label: 'Frequency', data: sortedSigns.map(s => s[1]),
                    backgroundColor: sortedSigns.map((_, i) => `hsl(${250 + i * 8}, 70%, ${55 + i * 2}%)`),
                    borderRadius: 6, borderSkipped: false
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: textColor, font: { weight: '700' } }, grid: { display: false } },
                    y: { ticks: { color: textColor }, grid: { color: gridColor }, beginAtZero: true }
                }
            }
        });
    }

    // ===== UTILITY FUNCTIONS =====
    function getMostFrequent(arr) {
        if (!arr || arr.length === 0) return null;
        const freq = {};
        arr.forEach(item => freq[item] = (freq[item] || 0) + 1);
        return Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0];
    }

    function showToast(message, type = 'info') {
        document.querySelectorAll('.toast').forEach(t => t.remove());
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        const icons = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle' };
        toast.innerHTML = `<i class="fas ${icons[type]}"></i> ${message}`;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(20px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    function seedDemoData() {
        if (state.sessionData.length === 0) {
            state.sessionData = [
                { id: Date.now() - 300000, timestamp: new Date(Date.now() - 300000).toISOString(), signs: ['H', 'E', 'L', 'L', 'O'], emotions: ['Happy', 'Happy', 'Neutral', 'Happy', 'Surprise'], sentence: 'HELLO', interpretation: 'The user is greeting with a happy demeanor.' },
                { id: Date.now() - 600000, timestamp: new Date(Date.now() - 600000).toISOString(), signs: ['T', 'H', 'A', 'N', 'K', 'Y', 'O', 'U'], emotions: ['Happy', 'Happy', 'Happy', 'Neutral'], sentence: 'THANKYOU', interpretation: 'The user is expressing gratitude.' },
                { id: Date.now() - 900000, timestamp: new Date(Date.now() - 900000).toISOString(), signs: ['H', 'E', 'L', 'P'], emotions: ['Fear', 'Sad', 'Neutral', 'Fear'], sentence: 'HELP', interpretation: 'The user appears to be requesting assistance.' }
            ];
            localStorage.setItem('signai_sessions', JSON.stringify(state.sessionData));
        }
    }

    function init() {
        seedDemoData();
        initNavigation();
        initTheme();
        initMobileMenu();
        initNavScroll();
        initDetection();
        initLearnPage();
        initTranscriptActions();
        renderASLGrid();

        const hash = window.location.hash.replace('#', '');
        if (hash && $(`#page-${hash}`)) navigateTo(hash);

        console.log('%c🤟 SignAI Initialized', 'color: #6C63FF; font-size: 14px; font-weight: bold;');
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();

})();
