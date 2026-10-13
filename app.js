import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// Everything runs on this device: MediaPipe's WASM runtime and the model file are
// static downloads, and no camera frames are sent anywhere.
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Playable range: A2 (110 Hz) to A6 (1760 Hz), four octaves.
const MIN_MIDI = 45;
const MAX_MIDI = 93;

// Vertical band of the camera frame that maps to the full range (0 = top, 1 = bottom).
const PLAY_TOP = 0.12;
const PLAY_BOTTOM = 0.88;

// How long a hand may drop out of tracking before we treat it as gone.
const HAND_GRACE_MS = 300;

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// Allowed pitch classes per scale (0 = C). "smooth" means no snapping.
const SCALES = {
  smooth: null,
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  major: [0, 2, 4, 5, 7, 9, 11],
  pentatonic: [9, 0, 2, 4, 7],
};

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const $ = (id) => document.getElementById(id);
const video = $("video");
const bg = $("bg");
const fx = $("fx");
const bgCtx = bg.getContext("2d");
const fxCtx = fx.getContext("2d");

const ui = {
  status: $("status"),
  start: $("start"),
  intro: $("intro"),
  error: $("error"),
  wave: $("wave"),
  scale: $("scale"),
  showCam: $("showCam"),
  meterFill: $("meterFill"),
  note: $("note"),
  freq: $("freq"),
};

// ---------- Small helpers ----------

const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const lerp = (a, b, t) => a + (b - a) * t;
const midiToFreq = (m) => 440 * Math.pow(2, (m - 69) / 12);
const pitchHue = (midi) => (((midi % 12) + 12) % 12) / 12 * 360;

function noteLabel(midi) {
  const rounded = Math.round(midi);
  return NOTE_NAMES[((rounded % 12) + 12) % 12] + (Math.floor(rounded / 12) - 1);
}

function snapMidi(midi, scaleName) {
  const pcs = SCALES[scaleName];
  if (!pcs) return midi;
  let best = midi;
  let bestDist = Infinity;
  for (let m = Math.floor(midi) - 2; m <= Math.ceil(midi) + 2; m++) {
    if (!pcs.includes(((m % 12) + 12) % 12)) continue;
    const d = Math.abs(m - midi);
    if (d < bestDist) { bestDist = d; best = m; }
  }
  return best;
}

// Hand height in camera space (y: 0 top, 1 bottom) -> 0..1, higher hand = larger value.
const heightNorm = (y) => clamp((PLAY_BOTTOM - y) / (PLAY_BOTTOM - PLAY_TOP));
const normToY = (n) => PLAY_BOTTOM - n * (PLAY_BOTTOM - PLAY_TOP);

// ---------- Audio engine ----------

function createAudio() {
  const ctx = new AudioContext();

  const osc1 = ctx.createOscillator();
  const osc2 = ctx.createOscillator();
  osc1.type = ui.wave.value;
  osc2.type = "sine";
  osc2.detune.value = 6; // slight chorus

  const g1 = ctx.createGain(); g1.gain.value = 0.55;
  const g2 = ctx.createGain(); g2.gain.value = 0.45;

  // Vibrato: an LFO modulating both oscillators' detune (in cents).
  const lfo = ctx.createOscillator();
  lfo.frequency.value = 5.5;
  const lfoDepth = ctx.createGain();
  lfoDepth.gain.value = 0;
  lfo.connect(lfoDepth);
  lfoDepth.connect(osc1.detune);
  lfoDepth.connect(osc2.detune);

  const filter = ctx.createBiquadFilter();
  filter.type = "lowpass";
  filter.Q.value = 3;
  filter.frequency.value = 2000;

  const master = ctx.createGain();
  master.gain.value = 0;

  // Feedback echo for a bit of space.
  const delay = ctx.createDelay(1);
  delay.delayTime.value = 0.28;
  const feedback = ctx.createGain(); feedback.gain.value = 0.32;
  const wet = ctx.createGain(); wet.gain.value = 0.28;

  const comp = ctx.createDynamicsCompressor();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.8;

  osc1.connect(g1).connect(filter);
  osc2.connect(g2).connect(filter);
  filter.connect(master);
  master.connect(comp);
  master.connect(delay);
  delay.connect(feedback).connect(delay);
  delay.connect(wet).connect(comp);
  comp.connect(analyser).connect(ctx.destination);

  osc1.start();
  osc2.start();
  lfo.start();

  return {
    ctx,
    analyser,
    freqData: new Uint8Array(analyser.frequencyBinCount),
    waveData: new Float32Array(analyser.fftSize),
    setWave(type) { osc1.type = type; },
    apply({ freq, volume, brightness, vibrato, glide }) {
      const t = ctx.currentTime;
      osc1.frequency.setTargetAtTime(freq, t, glide);
      osc2.frequency.setTargetAtTime(freq, t, glide);
      // Squared for a more natural loudness curve.
      master.gain.setTargetAtTime(volume * volume * 0.6, t, 0.04);
      filter.frequency.setTargetAtTime(Math.max(freq * 1.5, 300 * Math.pow(2, brightness * 5.5)), t, 0.06);
      lfoDepth.gain.setTargetAtTime(vibrato * 45, t, 0.08);
    },
  };
}

// ---------- Hand tracking ----------

let landmarker = null;

async function loadLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(vision, options("GPU"));
  } catch {
    return await HandLandmarker.createFromOptions(vision, options("CPU"));
  }
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

// Convert MediaPipe landmarks into mirrored ("selfie view") coordinates plus gesture metrics.
function readHands(result) {
  const aspect = video.videoWidth / video.videoHeight || 16 / 9;
  const dist = (a, b) => Math.hypot((a.x - b.x) * aspect, a.y - b.y);

  return (result.landmarks || []).map((lm) => {
    const pts = lm.map((p) => ({ x: 1 - p.x, y: p.y }));
    const palm = [0, 5, 9, 13, 17].reduce(
      (acc, i) => ({ x: acc.x + pts[i].x / 5, y: acc.y + pts[i].y / 5 }),
      { x: 0, y: 0 },
    );
    const size = dist(pts[0], pts[9]) || 1e-3;
    const pinch = dist(pts[4], pts[8]) / size;
    const openness = [8, 12, 16, 20].reduce((s, i) => s + dist(pts[i], pts[0]), 0) / 4 / size;
    return { pts, palm, pinch, openness };
  });
}

// ---------- State ----------

let audio = null;
let hands = [];
let pitchHand = null;
let volumeHand = null;
let lastVideoTime = -1;
let lastPitchSeen = -Infinity;
let lastVolumeSeen = -Infinity;
let cameraOn = false;

const control = { pitchNorm: 0.5, volume: 0, brightness: 0.6, vibrato: 0 };
const sound = { midi: 69, displayMidi: 69, volume: 0, displayVolume: 0, brightness: 0.6, vibrato: 0 };
const pointer = { down: false, x: 0.5, y: 0.5 };
const particles = [];

// Screen layout of the (mirrored, cover-fitted) camera frame.
const view = { w: 0, h: 0, dpr: 1, ox: 0, oy: 0, dw: 0, dh: 0 };

function resize() {
  view.dpr = Math.min(window.devicePixelRatio || 1, 2);
  view.w = window.innerWidth;
  view.h = window.innerHeight;
  for (const c of [bg, fx]) {
    c.width = Math.round(view.w * view.dpr);
    c.height = Math.round(view.h * view.dpr);
  }
  bgCtx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  fxCtx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  layoutVideo();
}

function layoutVideo() {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const scale = Math.max(view.w / vw, view.h / vh);
  view.dw = vw * scale;
  view.dh = vh * scale;
  view.ox = (view.w - view.dw) / 2;
  view.oy = (view.h - view.dh) / 2;
}

const toScreen = (p) => ({ x: view.ox + p.x * view.dw, y: view.oy + p.y * view.dh });
const screenToCamY = (sy) => (sy - view.oy) / view.dh;

// ---------- Control logic ----------

function updateControl(now) {
  // The hand further right on screen plays pitch; the other one sets volume.
  const sorted = [...hands].sort((a, b) => b.palm.x - a.palm.x);
  pitchHand = null;
  volumeHand = null;
  if (sorted.length >= 2) {
    [pitchHand, volumeHand] = sorted;
  } else if (sorted.length === 1) {
    if (sorted[0].palm.x >= 0.5) pitchHand = sorted[0];
    else volumeHand = sorted[0];
  }

  if (pitchHand) {
    lastPitchSeen = now;
    control.pitchNorm = heightNorm(pitchHand.pts[8].y); // index fingertip
    control.vibrato = clamp((0.5 - pitchHand.pinch) / 0.3);
  }
  if (volumeHand) {
    lastVolumeSeen = now;
    control.volume = clamp((heightNorm(volumeHand.palm.y) - 0.05) / 0.85);
    control.brightness = clamp(volumeHand.openness - 1.0);
  }

  const pitchActive = now - lastPitchSeen < HAND_GRACE_MS;
  const volumeActive = now - lastVolumeSeen < HAND_GRACE_MS;

  let volume;
  let brightness;
  if (!pitchActive && !volumeActive && pointer.down) {
    // Mouse / touch fallback: vertical = pitch, horizontal = brightness.
    control.pitchNorm = heightNorm(screenToCamY(pointer.y * view.h));
    control.vibrato = 0;
    volume = 0.7;
    brightness = pointer.x;
  } else {
    // With only the pitch hand visible, play at a comfortable fixed volume.
    volume = volumeActive ? control.volume : pitchActive ? 0.6 : 0;
    brightness = volumeActive ? control.brightness : 0.6;
    if (!pitchActive) control.vibrato = 0;
  }

  const rawMidi = MIN_MIDI + control.pitchNorm * (MAX_MIDI - MIN_MIDI);
  sound.midi = snapMidi(rawMidi, ui.scale.value);
  sound.volume = volume;
  sound.brightness = brightness;
  sound.vibrato = control.vibrato;

  sound.displayMidi = lerp(sound.displayMidi, sound.midi, 0.3);
  sound.displayVolume = lerp(sound.displayVolume, sound.volume, 0.2);

  audio.apply({
    freq: midiToFreq(sound.midi),
    volume: sound.volume,
    brightness: sound.brightness,
    vibrato: sound.vibrato,
    glide: ui.scale.value === "smooth" ? 0.04 : 0.015,
  });
}

// ---------- Drawing: background layer (camera, guides, skeletons) ----------

function drawBackground() {
  const c = bgCtx;
  c.fillStyle = "#05050f";
  c.fillRect(0, 0, view.w, view.h);

  if (cameraOn && ui.showCam.checked && video.readyState >= 2) {
    c.save();
    c.globalAlpha = 0.35;
    c.translate(view.w, 0);
    c.scale(-1, 1);
    c.drawImage(video, view.ox, view.oy, view.dw, view.dh);
    c.restore();
  }

  drawPitchGuide(c);
  if (pitchHand) drawSkeleton(c, pitchHand, "94, 231, 255");
  if (volumeHand) drawSkeleton(c, volumeHand, "255, 94, 200");
}

function drawPitchGuide(c) {
  const pcs = SCALES[ui.scale.value];
  const x0 = view.w * 0.62;
  const x1 = view.w - 12;
  const current = Math.round(sound.midi);

  c.save();
  c.font = "11px system-ui, sans-serif";
  c.textAlign = "right";
  c.textBaseline = "middle";
  for (let m = MIN_MIDI; m <= MAX_MIDI; m++) {
    const pc = m % 12;
    const inScale = pcs ? pcs.includes(pc) : !NOTE_NAMES[pc].includes("#");
    if (!inScale) continue;
    const y = view.oy + normToY((m - MIN_MIDI) / (MAX_MIDI - MIN_MIDI)) * view.dh;
    if (y < 0 || y > view.h) continue;
    const isCurrent = m === current && sound.volume > 0.02;
    c.strokeStyle = isCurrent ? `hsla(${pitchHue(m)}, 90%, 65%, 0.9)` : "rgba(255,255,255,0.07)";
    c.lineWidth = isCurrent ? 2 : 1;
    c.beginPath();
    c.moveTo(x0, y);
    c.lineTo(x1 - 34, y);
    c.stroke();
    c.fillStyle = isCurrent ? "#fff" : "rgba(255,255,255,0.3)";
    c.fillText(noteLabel(m), x1, y);
  }
  c.restore();
}

function drawSkeleton(c, hand, rgb) {
  const pts = hand.pts.map(toScreen);
  c.save();
  c.strokeStyle = `rgba(${rgb}, 0.75)`;
  c.lineWidth = 2;
  c.lineCap = "round";
  c.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    c.moveTo(pts[a].x, pts[a].y);
    c.lineTo(pts[b].x, pts[b].y);
  }
  c.stroke();
  c.fillStyle = `rgba(${rgb}, 1)`;
  for (const p of pts) {
    c.beginPath();
    c.arc(p.x, p.y, 3, 0, Math.PI * 2);
    c.fill();
  }
  c.restore();
}

// ---------- Drawing: audio-reactive effects layer (with trails) ----------

function drawEffects(now) {
  const c = fxCtx;
  const { analyser, freqData, waveData } = audio;
  analyser.getByteFrequencyData(freqData);
  analyser.getFloatTimeDomainData(waveData);

  // Fade previous frame to leave light trails.
  c.globalCompositeOperation = "destination-out";
  c.fillStyle = "rgba(0, 0, 0, 0.18)";
  c.fillRect(0, 0, view.w, view.h);
  c.globalCompositeOperation = "lighter";

  const hue = pitchHue(sound.displayMidi);
  const vol = sound.displayVolume;
  const cx = view.w / 2;
  const cy = view.h / 2;
  const base = Math.min(view.w, view.h) * 0.16;

  let bass = 0;
  for (let i = 0; i < 16; i++) bass += freqData[i];
  bass /= 16 * 255;
  const radius = base * (1 + bass * 0.5);

  // Radial spectrum: mirrored around the circle so it looks symmetric.
  const bars = 96;
  const usable = Math.floor(freqData.length * 0.35);
  for (let i = 0; i < bars; i++) {
    const v = freqData[Math.floor((i / bars) * usable)] / 255;
    if (v < 0.02) continue;
    const len = v * base * 1.4;
    for (const side of [1, -1]) {
      const a = -Math.PI / 2 + side * (i / bars) * Math.PI;
      const cos = Math.cos(a);
      const sin = Math.sin(a);
      c.strokeStyle = `hsla(${(hue + i * 1.5) % 360}, 90%, ${45 + v * 25}%, ${0.25 + v * 0.6})`;
      c.lineWidth = 3;
      c.beginPath();
      c.moveTo(cx + cos * radius, cy + sin * radius);
      c.lineTo(cx + cos * (radius + len), cy + sin * (radius + len));
      c.stroke();
    }
  }

  // Waveform ring.
  c.strokeStyle = `hsla(${hue}, 100%, 70%, ${0.3 + vol * 0.7})`;
  c.lineWidth = 2;
  c.beginPath();
  const step = 4;
  const n = Math.floor(waveData.length / step);
  for (let i = 0; i <= n; i++) {
    const s = waveData[(i * step) % waveData.length];
    const a = (i / n) * Math.PI * 2;
    const r = radius * 0.85 + s * base * 0.9;
    const x = cx + Math.cos(a) * r;
    const y = cy + Math.sin(a) * r;
    if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
  }
  c.closePath();
  c.stroke();

  // Glowing orb at the source of the sound (pitch fingertip or pointer).
  let source = null;
  if (pitchHand) source = toScreen(pitchHand.pts[8]);
  else if (pointer.down) source = { x: pointer.x * view.w, y: pointer.y * view.h };

  if (source && vol > 0.01) {
    const wobble = sound.vibrato * 6 * Math.sin(now / 30);
    const orbR = 10 + vol * 30 + bass * 20;
    const grad = c.createRadialGradient(source.x + wobble, source.y, 0, source.x + wobble, source.y, orbR * 2.5);
    grad.addColorStop(0, `hsla(${hue}, 100%, 85%, ${0.8 * vol + 0.2})`);
    grad.addColorStop(0.4, `hsla(${hue}, 100%, 60%, ${0.4 * vol})`);
    grad.addColorStop(1, `hsla(${hue}, 100%, 50%, 0)`);
    c.fillStyle = grad;
    c.beginPath();
    c.arc(source.x + wobble, source.y, orbR * 2.5, 0, Math.PI * 2);
    c.fill();

    // Emit particles in proportion to volume.
    const count = Math.round(vol * 5 + bass * 4);
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2;
      const speed = 1 + Math.random() * 3 * (0.5 + vol);
      particles.push({
        x: source.x, y: source.y,
        vx: Math.cos(a) * speed, vy: Math.sin(a) * speed - 0.5,
        life: 1,
        decay: 0.01 + Math.random() * 0.02,
        hue: (hue + (Math.random() - 0.5) * 40 + 360) % 360,
        size: 1.5 + Math.random() * 3,
      });
    }
  }

  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vx *= 0.985;
    p.vy = p.vy * 0.985 - 0.02; // gentle upward drift
    p.life -= p.decay;
    if (p.life <= 0) { particles.splice(i, 1); continue; }
    c.fillStyle = `hsla(${p.hue}, 100%, 65%, ${p.life})`;
    c.beginPath();
    c.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
    c.fill();
  }
  if (particles.length > 1500) particles.splice(0, particles.length - 1500);

  c.globalCompositeOperation = "source-over";
}

function updateHud() {
  const playing = sound.volume > 0.02;
  ui.note.textContent = playing ? noteLabel(sound.midi) : "–";
  ui.note.style.color = playing ? `hsl(${pitchHue(sound.midi)}, 90%, 70%)` : "var(--muted)";
  ui.freq.textContent = playing ? `${midiToFreq(sound.midi).toFixed(1)} Hz` : "raise your hands";
  ui.meterFill.style.height = `${Math.round(sound.displayVolume * 100)}%`;
}

// ---------- Main loop ----------

function frame(now) {
  requestAnimationFrame(frame);

  if (cameraOn && landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    hands = readHands(landmarker.detectForVideo(video, now));
  }

  updateControl(now);
  drawBackground();
  drawEffects(now);
  updateHud();
}

// ---------- Wiring ----------

const landmarkerReady = loadLandmarker()
  .then((lm) => {
    landmarker = lm;
    ui.status.textContent = "Hand model ready";
  })
  .catch((err) => {
    console.error(err);
    ui.status.textContent = "Hand model failed to load: mouse mode only";
  })
  .finally(() => {
    ui.start.disabled = false;
    ui.start.textContent = "Start";
  });

ui.start.addEventListener("click", async () => {
  ui.start.disabled = true;
  ui.start.textContent = "Starting…";

  // The audio context must be created inside a user gesture.
  audio = createAudio();
  await landmarkerReady;

  try {
    await startCamera();
    cameraOn = true;
    layoutVideo();
    ui.status.textContent = landmarker
      ? "Tracking: right hand = pitch, left hand = volume"
      : "Camera on, but hand tracking is unavailable: drag to play";
  } catch (err) {
    console.error(err);
    ui.status.textContent = "Camera unavailable: click and drag to play";
  }

  ui.intro.classList.add("hidden");
  requestAnimationFrame(frame);
});

ui.wave.addEventListener("change", () => audio?.setWave(ui.wave.value));

function setPointer(e) {
  pointer.x = clamp(e.clientX / view.w);
  pointer.y = clamp(e.clientY / view.h);
}

fx.addEventListener("pointerdown", (e) => {
  pointer.down = true;
  setPointer(e);
  fx.setPointerCapture(e.pointerId);
});
fx.addEventListener("pointermove", setPointer);
for (const type of ["pointerup", "pointercancel"]) {
  fx.addEventListener(type, () => { pointer.down = false; });
}

window.addEventListener("resize", resize);
video.addEventListener("loadedmetadata", layoutVideo);
resize();
