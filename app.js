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
  psychedelic: $("psychedelic"),
  meterFill: $("meterFill"),
  note: $("note"),
  freq: $("freq"),
  tutorialBtn: $("tutorialBtn"),
  tutorial: $("tutorial"),
  tutStep: $("tutStep"),
  tutTitle: $("tutTitle"),
  tutText: $("tutText"),
  tutBar: $("tutBar"),
  tutSkip: $("tutSkip"),
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
  for (const c of [bg, fx, trail]) {
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

// ---------- Tutorial ----------

const TUTORIAL_KEY = "air-theremin-tutorial-done";

// Each step highlights a zone on screen and completes when its check passes.
const TUTORIAL_STEPS = [
  {
    zone: "pitch", gesture: "hold",
    title: "Raise your right hand",
    text: "Hold it up inside the blue box, palm facing the camera, about an arm's length away.",
    check: (now) => tutorial.hold(!!pitchHand, now, 800),
  },
  {
    zone: "pitch", gesture: "sweep",
    title: "Move it up and down",
    text: "Higher hand, higher note. Sweep slowly from the bottom of the box to the top.",
    check: () => tutorial.sweep(pitchHand ? control.pitchNorm : null, 0.5),
  },
  {
    zone: "volume", gesture: "hold",
    title: "Now raise your left hand",
    text: "Keep your right hand up and bring your left hand into the pink box.",
    check: (now) => tutorial.hold(!!volumeHand, now, 800),
  },
  {
    zone: "volume", gesture: "sweep",
    title: "Your left hand sets the volume",
    text: "Lift it to get louder, lower it to fade out.",
    check: () => tutorial.sweep(volumeHand ? control.volume : null, 0.5),
  },
  {
    zone: "pitch", gesture: "pinch",
    title: "Pinch for vibrato",
    text: "Touch your right thumb and index finger together to make the note wobble.",
    check: (now) => tutorial.hold(!!pitchHand && control.vibrato > 0.6, now, 500),
  },
];

const tutorial = {
  active: false,
  step: 0,
  progress: 0,
  holdSince: null,
  lo: Infinity,
  hi: -Infinity,
  finishedAt: null,

  // Condition must stay true for `ms` milliseconds.
  hold(cond, now, ms) {
    if (!cond) { this.holdSince = null; this.progress = 0; return false; }
    this.holdSince ??= now;
    this.progress = clamp((now - this.holdSince) / ms);
    return this.progress >= 1;
  },

  // Value must cover at least `span` of its 0..1 range.
  sweep(value, span) {
    if (value == null) return false;
    this.lo = Math.min(this.lo, value);
    this.hi = Math.max(this.hi, value);
    this.progress = clamp((this.hi - this.lo) / span);
    return this.progress >= 1;
  },
};

function startTutorial() {
  Object.assign(tutorial, { active: true, finishedAt: null });
  goToStep(0);
  ui.tutorial.classList.remove("hidden");
}

function goToStep(i) {
  Object.assign(tutorial, { step: i, progress: 0, holdSince: null, lo: Infinity, hi: -Infinity });
  const step = TUTORIAL_STEPS[i];
  ui.tutorial.dataset.zone = step.zone;
  ui.tutStep.textContent = `Step ${i + 1} of ${TUTORIAL_STEPS.length}`;
  ui.tutTitle.textContent = step.title;
  ui.tutText.textContent = step.text;
  ui.tutSkip.textContent = "Skip tutorial";
}

function finishTutorial(now) {
  tutorial.finishedAt = now;
  tutorial.progress = 1;
  ui.tutorial.dataset.zone = "done";
  ui.tutStep.textContent = "Tutorial complete";
  ui.tutTitle.textContent = "You're ready. Play away!";
  ui.tutText.textContent = "Replay this anytime with “How to play”.";
  ui.tutSkip.textContent = "Close";
  try { localStorage.setItem(TUTORIAL_KEY, "1"); } catch {}
}

function closeTutorial() {
  tutorial.active = false;
  ui.tutorial.classList.add("hidden");
  try { localStorage.setItem(TUTORIAL_KEY, "1"); } catch {}
}

function updateTutorial(now) {
  if (!tutorial.active) return;
  if (tutorial.finishedAt != null) {
    if (now - tutorial.finishedAt > 2500) closeTutorial();
    return;
  }
  if (TUTORIAL_STEPS[tutorial.step].check(now)) {
    if (tutorial.step + 1 < TUTORIAL_STEPS.length) goToStep(tutorial.step + 1);
    else finishTutorial(now);
  }
  ui.tutBar.style.width = `${Math.round(tutorial.progress * 100)}%`;
}

function tutorialWasSeen() {
  try { return localStorage.getItem(TUTORIAL_KEY) === "1"; } catch { return false; }
}

// Draws the target box for the current step plus a "ghost hand" showing the motion.
function drawTutorialZone(c, now) {
  if (!tutorial.active || tutorial.finishedAt != null) return;
  const step = TUTORIAL_STEPS[tutorial.step];
  const isPitch = step.zone === "pitch";
  const rgb = isPitch ? "94, 231, 255" : "255, 94, 200";
  const occupied = isPitch ? !!pitchHand : !!volumeHand;

  // Keep the box clear of the header and the tutorial card.
  const top = Math.max(view.oy + PLAY_TOP * view.dh, 90);
  const bottom = Math.max(top + 120, Math.min(view.oy + PLAY_BOTTOM * view.dh, view.h - 300));
  const width = Math.min(view.w * 0.34, 300);
  const cx = view.w * (isPitch ? 0.75 : 0.25);
  const x = cx - width / 2;
  const h = bottom - top;
  const pulse = 0.5 + 0.3 * Math.sin(now / 250);

  c.save();
  c.fillStyle = `rgba(${rgb}, ${occupied ? 0.12 : 0.05})`;
  c.strokeStyle = `rgba(${rgb}, ${occupied ? 0.9 : pulse})`;
  c.lineWidth = 2;
  if (!occupied) {
    c.setLineDash([10, 8]);
    c.lineDashOffset = -now / 40;
  }
  c.beginPath();
  c.roundRect(x, top, width, h, 18);
  c.fill();
  c.stroke();
  c.setLineDash([]);

  c.fillStyle = `rgba(${rgb}, 0.9)`;
  c.font = "600 13px system-ui, sans-serif";
  c.textAlign = "center";
  c.fillText(isPitch ? "RIGHT HAND · PITCH" : "LEFT HAND · VOLUME", cx, top - 10);
  if (step.gesture === "sweep") {
    c.font = "12px system-ui, sans-serif";
    c.fillText(isPitch ? "▲ higher" : "▲ louder", cx, top + 22);
    c.fillText(isPitch ? "▼ lower" : "▼ quieter", cx, bottom - 12);
  }

  // Ghost hand demonstrating the gesture.
  let gy = top + h / 2;
  let size = Math.min(72, h * 0.4);
  let glyph = "✋";
  if (step.gesture === "hold") gy += Math.sin(now / 400) * 6;
  if (step.gesture === "sweep") gy += Math.sin(now / 650) * (h / 2 - 50);
  if (step.gesture === "pinch") {
    glyph = Math.sin(now / 350) > 0 ? "🤏" : "✋";
    size *= 1.1;
  }
  c.globalAlpha = occupied ? 0.3 : 0.75;
  c.font = `${size}px system-ui, "Segoe UI Emoji", "Apple Color Emoji", sans-serif`;
  c.textBaseline = "middle";
  c.translate(cx, gy);
  if (!isPitch) c.scale(-1, 1); // mirror the emoji so it reads as a left hand
  c.fillText(glyph, 0, 0);
  c.restore();
}

// ---------- Drawing: background layer (camera, guides, skeletons) ----------

function drawBackground() {
  const c = bgCtx;
  const psy = psychedelic();
  c.fillStyle = "#05050f";
  c.fillRect(0, 0, view.w, view.h);

  if (psy) {
    // Color wash that cycles hue and glows brighter as the sound gets louder.
    const cx = view.w / 2;
    const cy = view.h / 2;
    const h = (pitchHue(sound.displayMidi) + vis.hueShift) % 360;
    const g = c.createRadialGradient(cx, cy, 0, cx, cy, Math.hypot(cx, cy));
    g.addColorStop(0, `hsl(${h}, 80%, ${5 + vis.energy * 18}%)`);
    g.addColorStop(0.55, `hsl(${(h + 120) % 360}, 70%, ${3 + vis.energy * 8}%)`);
    g.addColorStop(1, "#05050f");
    c.fillStyle = g;
    c.fillRect(0, 0, view.w, view.h);
  }

  if (cameraOn && ui.showCam.checked && video.readyState >= 2) {
    c.save();
    c.globalAlpha = psy ? 0.25 + vis.energy * 0.25 : 0.35;
    if (psy) c.filter = `hue-rotate(${vis.hueShift}deg) saturate(${2 + vis.energy * 3}) contrast(1.25)`;
    c.translate(view.w, 0);
    c.scale(-1, 1);
    c.drawImage(video, view.ox, view.oy, view.dw, view.dh);
    c.restore();
  }

  drawPitchGuide(c);
  drawTutorialZone(c, performance.now());
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

// ---------- Drawing: audio-reactive effects layer ----------

// Loudness-driven state shared by both layers.
const vis = { energy: 0, avg: 0, bass: 0, hueShift: 0, spin: 0, lastRing: 0, rings: [] };

const KALEIDO_ARMS = 6;
const MAX_PARTICLES = 450;

// Off-screen copy of the previous frame, redrawn zoomed and rotated for the tunnel effect.
const trail = document.createElement("canvas");
const trailCtx = trail.getContext("2d");

const psychedelic = () => ui.psychedelic.checked;

// Measure what is actually coming out of the speakers (including the echo tail).
function measureAudio(now) {
  const { analyser, freqData, waveData } = audio;
  analyser.getByteFrequencyData(freqData);
  analyser.getFloatTimeDomainData(waveData);

  let sum = 0;
  for (let i = 0; i < waveData.length; i++) sum += waveData[i] * waveData[i];
  const level = clamp(Math.sqrt(sum / waveData.length) * 3.5);

  // Fast attack, slow release, so swells hit instantly and fade smoothly.
  vis.energy = lerp(vis.energy, level, level > vis.energy ? 0.5 : 0.08);

  let bass = 0;
  for (let i = 0; i < 16; i++) bass += freqData[i];
  vis.bass = bass / (16 * 255);

  // A sudden jump in loudness above the running average fires a shockwave ring.
  if (level - vis.avg > 0.15 && now - vis.lastRing > 180) {
    vis.lastRing = now;
    vis.rings.push({ r: Math.min(view.w, view.h) * 0.1, life: 1, hue: vis.hueShift + pitchHue(sound.midi) });
  }
  vis.avg = lerp(vis.avg, level, 0.05);

  const psy = psychedelic();
  vis.hueShift = (vis.hueShift + (psy ? 0.4 + vis.energy * 3.5 : 0)) % 360;
  vis.spin += 0.002 + vis.energy * (psy ? 0.03 : 0.004);
}

// Run `draw` once per kaleidoscope segment, rotated (and alternately mirrored) around the center.
// Segment 0 is the identity, so the original stays under the fingertip.
function kaleidoscope(c, arms, draw) {
  const cx = view.w / 2;
  const cy = view.h / 2;
  for (let k = 0; k < arms; k++) {
    c.save();
    c.translate(cx, cy);
    c.rotate((k / arms) * Math.PI * 2);
    if (k % 2 === 1) c.scale(-1, 1);
    c.translate(-cx, -cy);
    draw();
    c.restore();
  }
}

function drawEffects(now) {
  const c = fxCtx;
  const { freqData, waveData } = audio;
  const psy = psychedelic();
  const e = vis.energy;
  const hue = (pitchHue(sound.displayMidi) + vis.hueShift) % 360;
  const cx = view.w / 2;
  const cy = view.h / 2;
  const base = Math.min(view.w, view.h) * 0.16;

  if (psy) {
    // Feedback tunnel: last frame comes back slightly zoomed and twisted. Louder = faster, longer trails.
    trailCtx.setTransform(1, 0, 0, 1, 0, 0);
    trailCtx.clearRect(0, 0, trail.width, trail.height);
    trailCtx.drawImage(fx, 0, 0);

    c.save();
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, fx.width, fx.height);
    c.restore();

    c.save();
    c.globalAlpha = 0.84 + e * 0.12;
    c.translate(cx, cy);
    c.rotate(0.006 + e * 0.025 + sound.vibrato * 0.03);
    const zoom = 1.008 + e * 0.045;
    c.scale(zoom, zoom);
    c.translate(-cx, -cy);
    c.drawImage(trail, 0, 0, view.w, view.h);
    c.restore();
  } else {
    c.globalCompositeOperation = "destination-out";
    c.fillStyle = "rgba(0, 0, 0, 0.18)";
    c.fillRect(0, 0, view.w, view.h);
  }

  c.globalCompositeOperation = "lighter";
  const radius = base * (1 + vis.bass * 0.5 + (psy ? e * 0.6 : 0));

  // Center bloom that breathes with loudness.
  if (psy && e > 0.02) {
    const g = c.createRadialGradient(cx, cy, 0, cx, cy, radius * (1.2 + e * 2));
    g.addColorStop(0, `hsla(${hue}, 100%, 70%, ${e * 0.35})`);
    g.addColorStop(0.5, `hsla(${(hue + 60) % 360}, 100%, 50%, ${e * 0.15})`);
    g.addColorStop(1, "hsla(0, 0%, 0%, 0)");
    c.fillStyle = g;
    c.fillRect(0, 0, view.w, view.h);
  }

  // Radial spectrum: a full rainbow wheel that spins faster when louder.
  const bars = psy ? 120 : 96;
  const usable = Math.floor(freqData.length * 0.35);
  c.lineCap = "round";
  for (let i = 0; i < bars; i++) {
    const v = freqData[Math.floor(((i < bars / 2 ? i : bars - i) / (bars / 2)) * usable)] / 255;
    if (v < 0.02) continue;
    const len = v * base * (psy ? 1.2 + e * 1.8 : 1.4);
    const a = (i / bars) * Math.PI * 2 + vis.spin;
    const barHue = psy ? (hue + (i / bars) * 360) % 360 : (hue + i * 1.5) % 360;
    c.strokeStyle = `hsla(${barHue}, 100%, ${45 + v * 25}%, ${0.25 + v * 0.6})`;
    c.lineWidth = psy ? 2 + e * 5 : 3;
    c.beginPath();
    c.moveTo(cx + Math.cos(a) * radius, cy + Math.sin(a) * radius);
    c.lineTo(cx + Math.cos(a) * (radius + len), cy + Math.sin(a) * (radius + len));
    c.stroke();
  }

  // Waveform mandala: several rotated copies of the live waveform.
  const rings = psy ? 3 : 1;
  const step = 4;
  const n = Math.floor(waveData.length / step);
  for (let j = 0; j < rings; j++) {
    const offset = (j / rings) * Math.PI * 2 + vis.spin * (j % 2 ? -1 : 1);
    const scale = 0.85 - j * 0.18;
    c.strokeStyle = `hsla(${(hue + j * 120) % 360}, 100%, 70%, ${0.3 + e * 0.7})`;
    c.lineWidth = psy ? 1.5 + e * 3 : 2;
    c.beginPath();
    for (let i = 0; i <= n; i++) {
      const s = waveData[(i * step) % waveData.length];
      const a = (i / n) * Math.PI * 2 + offset;
      const r = radius * scale + s * base * (psy ? 0.9 + e * 2 : 0.9);
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.closePath();
    c.stroke();
  }

  // Shockwave rings fired by sudden swells in volume.
  for (let i = vis.rings.length - 1; i >= 0; i--) {
    const ring = vis.rings[i];
    ring.r += 6 + e * 14;
    ring.life -= 0.02;
    if (ring.life <= 0) { vis.rings.splice(i, 1); continue; }
    c.strokeStyle = `hsla(${(ring.hue + (1 - ring.life) * 180) % 360}, 100%, 65%, ${ring.life * 0.8})`;
    c.lineWidth = 8 * ring.life;
    c.beginPath();
    c.arc(cx, cy, ring.r, 0, Math.PI * 2);
    c.stroke();
  }

  // Sound source: pitch fingertip or pointer.
  let source = null;
  if (pitchHand) source = toScreen(pitchHand.pts[8]);
  else if (pointer.down) source = { x: pointer.x * view.w, y: pointer.y * view.h };

  const vol = sound.displayVolume;
  if (source && vol > 0.01) {
    const count = Math.round((vol * 4 + e * 6) * (psy ? 1 : 0.8));
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2;
      const speed = 1 + Math.random() * 3 * (0.5 + e * 1.5);
      particles.push({
        x: source.x, y: source.y,
        vx: Math.cos(a) * speed, vy: Math.sin(a) * speed - 0.5,
        life: 1,
        decay: 0.012 + Math.random() * 0.02,
        hue: (hue + (Math.random() - 0.5) * 80 + 360) % 360,
        size: 1.5 + Math.random() * 3 * (1 + e),
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
    if (p.life <= 0) particles.splice(i, 1);
  }
  if (particles.length > MAX_PARTICLES) particles.splice(0, particles.length - MAX_PARTICLES);

  const drawSparks = () => {
    for (const p of particles) {
      c.fillStyle = `hsla(${(p.hue + vis.hueShift * 0.5) % 360}, 100%, 65%, ${p.life})`;
      c.beginPath();
      c.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
      c.fill();
    }
  };

  const drawOrb = () => {
    if (!source || vol <= 0.01) return;
    const wobble = sound.vibrato * 6 * Math.sin(now / 30);
    const orbR = 10 + vol * 30 + vis.bass * 20 + (psy ? e * 25 : 0);
    const ox = source.x + wobble;
    const grad = c.createRadialGradient(ox, source.y, 0, ox, source.y, orbR * 2.5);
    grad.addColorStop(0, `hsla(${hue}, 100%, 85%, ${0.8 * vol + 0.2})`);
    grad.addColorStop(0.4, `hsla(${(hue + 40) % 360}, 100%, 60%, ${0.4 * vol})`);
    grad.addColorStop(1, `hsla(${hue}, 100%, 50%, 0)`);
    c.fillStyle = grad;
    c.beginPath();
    c.arc(ox, source.y, orbR * 2.5, 0, Math.PI * 2);
    c.fill();
  };

  if (psy) {
    // Mirror the fingertip's sparks and orb around the center: the hand paints a mandala.
    kaleidoscope(c, KALEIDO_ARMS, () => { drawSparks(); drawOrb(); });
  } else {
    drawSparks();
    drawOrb();
  }

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
  updateTutorial(now);
  measureAudio(now);
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

  // The tutorial needs live hand tracking; show it automatically on the first visit.
  if (cameraOn && landmarker) {
    ui.tutorialBtn.disabled = false;
    if (!tutorialWasSeen()) startTutorial();
  }
});

ui.wave.addEventListener("change", () => audio?.setWave(ui.wave.value));
ui.tutorialBtn.addEventListener("click", startTutorial);
ui.tutSkip.addEventListener("click", closeTutorial);

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

// Respect the OS "reduce motion" setting: start with the calmer visuals.
if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) ui.psychedelic.checked = false;

window.addEventListener("resize", resize);
video.addEventListener("loadedmetadata", layoutVideo);
resize();
