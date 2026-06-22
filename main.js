// if you're reading this... you either have no life, or you're interested in how this works.
// most likely the former. either way, welcome. grab a coffee, this one's got some math in it.
//
// Hands + FFmpeg come from <script> tags in index.html, available as globals.

// the usual suspects
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileInput2 = document.getElementById("fileInput2");
const playBtn = document.getElementById("playBtn");
const resetBtn = document.getElementById("resetBtn");
const webcamBtn = document.getElementById("webcamBtn"); 
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const world = document.getElementById("world");
const worldEmpty = document.getElementById("worldEmpty");
const rayLine = document.getElementById("rayLine");
const hudAngle = document.getElementById("hudAngle");
const hudConf = document.getElementById("hudConf");
const legLeft = document.getElementById("legLeft");
const legRight = document.getElementById("legRight");
const armLeft = document.getElementById("armLeft");
const armRight = document.getElementById("armRight");
const processingEl = document.getElementById("processing");
const procTitle = document.getElementById("procTitle");
const procSub = document.getElementById("procSub");

// ffmpeg.wasm instance — loaded lazily on first video upload, not on page load,
// so the user isn't sitting around waiting for 6MB of wasm before they've even
// dropped a file yet.
let ffmpegInstance = null;
let ffmpegLoading = false;

// offscreen canvas used for the cropped/upscaled second detection pass —
// never shown to the user, just a scratchpad we feed back into mediapipe
const cropCanvas = document.createElement("canvas");
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });
const CROP_SIZE = 480; // upscale target — bigger than a typical small in-frame hand

let hands = null;        // the legacy Hands instance
let rafId = null;
let lastVideoTime = -1;

// the legacy @mediapipe/hands library is async/callback-based (you call
// hands.send(), results show up later via onResults), unlike the newer
// tasks-vision library which returned results synchronously. this flag
// tracks whether we're still waiting on a result, so we don't fire off a
// new detection before the last one has come back.
let detectionInFlight = false;
let pendingCropForInFlightDetection = null; // crop rect used for the in-flight request, if any

// remembers where the hand roughly was last frame, in normalized [0,1] video
// coordinates, so we can crop tightly around it on the next frame instead of
// re-scanning the whole image at low effective resolution every time
let lastHandBox = null; // {cx, cy, size} or null if we don't have a fix yet
let framesSinceFullScan = 999; // forces a full-frame scan on the first frame

// bot's world position, in px offset from the center of the navigation panel
let botPos = { x: 0, y: 0 };
let botAngleDeg = 0;     // what's currently rendered (smoothed)
let targetAngleDeg = 0;  // where the hand says we should be heading
let hasDirection = false;
let walkCycle = 0;       // just a clock for the leg-swing animation, purely cosmetic

const BOT_SPEED = 1.6; // px/frame. crank this up if the bot is too much of a homebody

// raw per-frame results from mediapipe are noisy — even on decent footage,
// individual frames jump around, occasionally miss the hand entirely, or
// briefly latch onto a low-confidence garbage angle. none of that is
// "the model is broken," it's just normal frame-to-frame noise that every
// hand-tracking demo you've ever seen smooths out before showing it to you.
// we weren't smoothing it before. now we are.
//
// angleBuffer holds the last few detected angles (as unit vectors, NOT raw
// degrees — averaging degrees directly breaks horribly near the -180/180
// wraparound point, e.g. averaging -179 and 179 naively gives 0, which is
// exactly backwards). we average the unit vectors instead and convert back
// to an angle at the end, which handles wraparound correctly for free.
const SMOOTH_WINDOW = 8;
let angleBuffer = []; // each entry: {x, y} unit vector

// hysteresis on "do we currently see a hand": require a few consecutive
// hits before declaring "tracking", and a few consecutive misses before
// declaring "no hand" — instead of flipping state on every single frame.
let consecutiveHits = 0;
let consecutiveMisses = 0;
const PRESENCE_ON_THRESHOLD = 2;   // frames of detection before trusting it
const PRESENCE_OFF_THRESHOLD = 10; // frames of nothing before giving up

// the 21-point hand skeleton, wired up the way mediapipe expects.
// (if you ever forget which number is which finger: thumb is 1-4, index 5-8,
// middle 9-12, ring 13-16, pinky 17-20. wrist is 0. you're welcome, future me.)
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];

let isWebcam = false;

function initModel() {
  try {
    // Hands is a global from the hands.js <script> tag in index.html, not
    // an import — that's just how this older library is distributed.
    hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    // modelComplexity IS the lite-vs-full accuracy knob that doesn't exist
    // in the newer tasks-vision API. 1 = full model: better landmark
    // precision, slower. that tradeoff is exactly what we want here —
    // we already slowed video playback down to make room for the extra
    // compute, so there's no reason not to spend it on accuracy.
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    hands.onResults(onHandsResults);
    setStatus("ready", "model ready — drop a video or start webcam");
  } catch (err) {
    console.error("model refused to load. typical.", err);
    setStatus("error", "model failed to load (check connection)");
  }
}

function setStatus(kind, text) {
  statusDot.className = "status-dot" + (kind === "ready" ? " ready" : kind === "error" ? " error" : "");
  statusText.textContent = text;
}

initModel();

// ---- file handling: drag it, drop it, or just click like a normal person ----

[fileInput, fileInput2].forEach(input => {
  input.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) loadVideoFile(file);
  });
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("video/")) loadVideoFile(file);
});

function loadVideoFile(file) {
  stopWebcam();
  dropzone.classList.add("hidden");
  playBtn.disabled = true;
  resetBtn.disabled = true;
  setStatus("", "fixing video orientation…");
  fixVideoRotation(file);
}

// loads ffmpeg.wasm if not already loaded, then processes the video to
// physically bake any rotation metadata into the actual pixel data.
async function fixVideoRotation(file) {
  showProcessing("Fixing video orientation…", "Loading ffmpeg (first time only, ~6MB)");

  try {
    if (!ffmpegInstance) {
      const { FFmpeg } = FFmpegWASM;
      const { toBlobURL } = FFmpegUtil;
      ffmpegInstance = new FFmpeg();

      await ffmpegInstance.load({
        coreURL: "/ffmpeg/ffmpeg-core.js",
        wasmURL: "/ffmpeg/ffmpeg-core.wasm",
      });
    }

    showProcessing("Fixing video orientation…", "Processing…");

    const inputName = "input.mp4";
    const inputData = new Uint8Array(await file.arrayBuffer());

    ffmpegInstance.on("log", ({ type, message }) => {
      console.log(`[ffmpeg:${type}]`, message);
    });

    console.log("[ffmpeg] writing file, size:", inputData.byteLength);
    await ffmpegInstance.writeFile(inputName, inputData);
    console.log("[ffmpeg] file written, running exec...");

    const ret = await ffmpegInstance.exec([
      "-i", inputName,
      "-vf", "transpose=1",
      "-c:a", "copy",
      "output.mp4"
    ]);
    console.log("[ffmpeg] exec returned:", ret);

    console.log("[ffmpeg] reading output...");
    const outputData = await ffmpegInstance.readFile("output.mp4");

    const outputBlob = new Blob([outputData], { type: "video/mp4" });
    const url = URL.createObjectURL(outputBlob);

    await ffmpegInstance.deleteFile(inputName);
    await ffmpegInstance.deleteFile("output.mp4");

    hideProcessing();
    setVideoSource(url);

  } catch (err) {
    console.error("ffmpeg rotation fix failed:", err);
    hideProcessing();
    console.warn("falling back to original video (no rotation fix applied)");
    const url = URL.createObjectURL(file);
    setVideoSource(url);
  }
}

function setVideoSource(url) {
  video.src = url;
  video.load();
  playBtn.disabled = false;
  resetBtn.disabled = false;
  resetBot();
  setStatus("ready", "video ready — press play");

  video.addEventListener("loadedmetadata", () => {
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
  }, { once: true });
}

// ---- webcam handling module ----

if (webcamBtn) {
  webcamBtn.addEventListener("click", () => {
    if (isWebcam) {
      stopWebcam();
      resetBot();
      setStatus("ready", "Webcam stopped. Drop a video or click webcam again.");
    } else {
      startWebcam();
    }
  });
}

async function startWebcam() {
  video.pause();
  if (rafId) cancelAnimationFrame(rafId);
  resetBot();
  showProcessing("Starting Webcam...", "Requesting hardware access privileges...");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { 
        width: { ideal: 640 }, 
        height: { ideal: 480 },
        facingMode: "user" 
      },
      audio: false
    });

    hideProcessing();
    
    // Clear old file references entirely so the visual hardware stream binds seamlessly
    video.src = "";
    video.removeAttribute("src");
    
    video.srcObject = stream;
    isWebcam = true;
    
    // Flip elements on the GPU layout engine so interactions feel like a real mirror
    video.style.transform = "scaleX(-1)";
    overlay.style.transform = "scaleX(-1)";
    
    playBtn.disabled = true; 
    playBtn.textContent = "⏸ Live Tracking";
    if (webcamBtn) webcamBtn.textContent = "📷 Stop Webcam";
    setStatus("ready", "Webcam live — tracking hand position...");

    video.addEventListener("loadedmetadata", () => {
      video.playbackRate = 1.0; // Live feeds run in real-time execution speeds
      video.play();
      overlay.width = video.clientWidth;
      overlay.height = video.clientHeight;
      predictLoop();
    }, { once: true });

  } catch (err) {
    console.error("Webcam runtime mapping initialization failed:", err);
    hideProcessing();
    setStatus("error", "Webcam access denied or unavailable.");
    isWebcam = false;
  }
}

function stopWebcam() {
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(track => track.stop());
    video.srcObject = null;
  }
  isWebcam = false;
  
  // Revert matrix scaling modifications
  video.style.transform = "";
  overlay.style.transform = "";
  
  playBtn.disabled = false;
  playBtn.textContent = "▶ Play";
  if (webcamBtn) webcamBtn.textContent = "📷 Use Webcam";
}

function showProcessing(title, sub) {
  procTitle.textContent = title;
  procSub.textContent = sub;
  processingEl.classList.add("visible");
}

function hideProcessing() {
  processingEl.classList.remove("visible");
}

window.addEventListener("resize", () => {
  if (video.videoWidth || video.srcObject) {
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
  }
});

// ---- playback controls ----
const PLAYBACK_RATE = 0.4;

playBtn.addEventListener("click", () => {
  if (isWebcam) return; 
  if (video.paused) {
    video.playbackRate = PLAYBACK_RATE;
    video.play();
    playBtn.textContent = "⏸ Pause (0.4x — for accuracy)";
    predictLoop();
  } else {
    video.pause();
    playBtn.textContent = "▶ Play";
    if (rafId) cancelAnimationFrame(rafId);
  }
});

video.addEventListener("ended", () => {
  playBtn.textContent = "▶ Play";
  if (rafId) cancelAnimationFrame(rafId);
});

resetBtn.addEventListener("click", () => {
  stopWebcam();
  video.pause();
  video.currentTime = 0;
  playBtn.textContent = "▶ Play";
  if (rafId) cancelAnimationFrame(rafId);
  resetBot();
  clearOverlay();
});

function resetBot() {
  botPos = { x: 0, y: 0 };
  botAngleDeg = 0;
  targetAngleDeg = 0;
  hasDirection = false;
  angleBuffer = [];
  consecutiveHits = 0;
  consecutiveMisses = 0;
  lastHandBox = null;
  framesSinceFullScan = 999;
  detectionInFlight = false;
  pendingCropForInFlightDetection = null;
  renderBot();
  worldEmpty.classList.remove("hidden");
  rayLine.setAttribute("opacity", 0);
  hudAngle.textContent = "—";
  hudConf.textContent = "—";
}

function clearOverlay() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

// ---- the main loop ----
function predictLoop() {
  if (!isWebcam && (video.paused || video.ended)) return;

  if (hands && !detectionInFlight) {
    if (isWebcam || video.currentTime !== lastVideoTime) {
      if (!isWebcam) lastVideoTime = video.currentTime;
      sendFrameForDetection();
    }
  }

  stepBot();
  rafId = requestAnimationFrame(predictLoop);
}

const FULL_SCAN_INTERVAL = 20;

function sendFrameForDetection() {
  if (!isWebcam && (!video.videoWidth || !video.videoHeight)) return;

  const vw = video.videoWidth || video.clientWidth;
  const vh = video.videoHeight || video.clientHeight;

  framesSinceFullScan++;
  const needsFullScan = !lastHandBox || framesSinceFullScan >= FULL_SCAN_INTERVAL;

  detectionInFlight = true;

  if (needsFullScan) {
    framesSinceFullScan = 0;
    pendingCropForInFlightDetection = null;
    hands.send({ image: video });
    return;
  }

  const crop = computeCropRect(lastHandBox, vw, vh);
  cropCanvas.width = CROP_SIZE;
  cropCanvas.height = CROP_SIZE;
  cropCtx.drawImage(
    video,
    crop.x, crop.y, crop.size, crop.size,  
    0, 0, CROP_SIZE, CROP_SIZE             
  );

  pendingCropForInFlightDetection = crop;
  hands.send({ image: cropCanvas });
}

function onHandsResults(results) {
  const crop = pendingCropForInFlightDetection;
  detectionInFlight = false;

  const landmarksList = results.multiHandLandmarks;
  const vw = video.videoWidth || video.clientWidth;
  const vh = video.videoHeight || video.clientHeight;

  if (landmarksList && landmarksList.length > 0) {
    lastHandBox = boundingBoxFromLandmarks(landmarksList[0], crop, vw, vh);
  } else if (crop) {
    framesSinceFullScan = FULL_SCAN_INTERVAL - 3;
  } else {
    lastHandBox = null;
  }

  drawLandmarks(landmarksList, crop);
  updateDirection(landmarksList, crop);
}

function computeCropRect(box, vw, vh) {
  if (!box) {
    const size = Math.min(vw, vh);
    return { x: (vw - size) / 2, y: (vh - size) / 2, size };
  }

  const PADDING_FACTOR = 2.6; 
  const cx = box.cx * vw;
  const cy = box.cy * vh;
  let size = box.size * Math.max(vw, vh) * PADDING_FACTOR;

  size = Math.max(size, Math.min(vw, vh) * 0.25);
  size = Math.min(size, Math.min(vw, vh));

  let x = cx - size / 2;
  let y = cy - size / 2;
  x = Math.max(0, Math.min(vw - size, x));
  y = Math.max(0, Math.min(vh - size, y));

  return { x, y, size };
}

function boundingBoxFromLandmarks(landmarks, crop, vw, vh) {
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of landmarks) {
    let x = p.x, y = p.y;
    if (crop) {
      x = (crop.x + x * crop.size) / vw;
      y = (crop.y + y * crop.size) / vh;
    }
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  return {
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
    size: Math.max(maxX - minX, maxY - minY)
  };
}

function remapPoint(p, crop, vw, vh) {
  if (!crop) return p;
  return {
    x: (crop.x + p.x * crop.size) / vw,
    y: (crop.y + p.y * crop.size) / vh
  };
}

function drawLandmarks(landmarksList, crop) {
  clearOverlay();
  if (!landmarksList || landmarksList.length === 0) return;

  const w = overlay.width, h = overlay.height;
  const vw = video.videoWidth || video.clientWidth;
  const vh = video.videoHeight || video.clientHeight;

  for (const rawLandmarks of landmarksList) {
    const landmarks = crop ? rawLandmarks.map(p => remapPoint(p, crop, vw, vh)) : rawLandmarks;

    ctx.strokeStyle = "rgba(93,255,196,0.85)";
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      const p1 = landmarks[a], p2 = landmarks[b];
      
      // CSS handles the visual mirror transform; canvas mappings use raw values
      const x1 = p1.x * w;
      const x2 = p2.x * w;
      
      ctx.beginPath();
      ctx.moveTo(x1, p1.y * h);
      ctx.lineTo(x2, p2.y * h);
      ctx.stroke();
    }
    landmarks.forEach((p, i) => {
      const px = p.x * w;
      
      ctx.beginPath();
      ctx.arc(px, p.y * h, i === 8 ? 5 : 3.2, 0, Math.PI * 2);
      ctx.fillStyle = i === 8 ? "#ffffff" : "#5dffc4"; 
      ctx.fill();
      if (i === 8) {
        ctx.strokeStyle = "rgba(93,255,196,0.9)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
  }
}

function updateDirection(landmarksList, crop) {
  const sawHand = landmarksList && landmarksList.length > 0;

  if (!sawHand) {
    consecutiveHits = 0;
    consecutiveMisses++;
    if (consecutiveMisses >= PRESENCE_OFF_THRESHOLD) {
      hasDirection = false;
      angleBuffer = [];
      hudConf.textContent = "no hand";
      rayLine.setAttribute("opacity", 0);
    }
    return;
  }

  consecutiveMisses = 0;
  consecutiveHits++;

  const vw = video.videoWidth || video.clientWidth;
  const vh = video.videoHeight || video.clientHeight;
  const rawLm = landmarksList[0];
  const base = remapPoint(rawLm[5], crop, vw, vh);
  const tip = remapPoint(rawLm[8], crop, vw, vh);

  // If webcam stream is active, invert the x vector difference to map mirrored screen space accurately
  const dx = isWebcam ? base.x - tip.x : tip.x - base.x;
  const dy = tip.y - base.y;

  const mag = Math.hypot(dx, dy) || 1;
  angleBuffer.push({ x: dx / mag, y: dy / mag });
  if (angleBuffer.length > SMOOTH_WINDOW) angleBuffer.shift();

  let sumX = 0, sumY = 0;
  for (const v of angleBuffer) { sumX += v.x; sumY += v.y; }
  const avgX = sumX / angleBuffer.length;
  const avgY = sumY / angleBuffer.length;

  const angleRad = Math.atan2(avgY, avgX);
  const angleDeg = angleRad * (180 / Math.PI);

  if (consecutiveHits < PRESENCE_ON_THRESHOLD) return;

  targetAngleDeg = angleDeg;
  hasDirection = true;
  worldEmpty.classList.add("hidden");

  hudAngle.textContent = `${angleDeg.toFixed(0)}°`;
  hudConf.textContent = "tracking";

  const worldRect = world.getBoundingClientRect();
  const cx = worldRect.width / 2;
  const cy = worldRect.height / 2;
  const len = Math.min(worldRect.width, worldRect.height) * 0.42;
  rayLine.setAttribute("x1", cx);
  rayLine.setAttribute("y1", cy);
  rayLine.setAttribute("x2", cx + Math.cos(angleRad) * len);
  rayLine.setAttribute("y2", cy + Math.sin(angleRad) * len);
  rayLine.setAttribute("opacity", 0.55);
}

// ---- move the little guy ----
function stepBot() {
  if (!hasDirection) return;

  let diff = targetAngleDeg - botAngleDeg;
  diff = ((diff + 180) % 360 + 360) % 360 - 180;
  botAngleDeg += diff * 0.12;

  const rad = botAngleDeg * (Math.PI / 180);
  const worldRect = world.getBoundingClientRect();
  const halfW = worldRect.width / 2;
  const halfH = worldRect.height / 2;

  let nx = botPos.x + Math.cos(rad) * BOT_SPEED;
  let ny = botPos.y + Math.sin(rad) * BOT_SPEED;

  const botHalfW = 36;
  const botHalfH = 39;
  if (nx > halfW - botHalfW) nx = -halfW + botHalfW;
  if (nx < -halfW + botHalfW) nx = halfW - botHalfW;
  if (ny > halfH - botHalfH) ny = -halfH + botHalfH;
  if (ny < -halfH + botHalfH) ny = halfH - botHalfH;

  botPos = { x: nx, y: ny };

  walkCycle += 0.25;
  renderBot();
}

function renderBot() {
  const svg = document.getElementById("botSvg");
  const visualOffset = 90;
  svg.style.transform =
    `translate(-50%, -50%) translate(${botPos.x}px, ${botPos.y}px) rotate(${botAngleDeg + visualOffset}deg)`;

  if (hasDirection) {
    const swing = Math.sin(walkCycle) * 12;
    legLeft.style.transform = `rotate(${swing}deg)`;
    legRight.style.transform = `rotate(${-swing}deg)`;
    armLeft.style.transform = `rotate(${-swing * 0.6}deg)`;
    armRight.style.transform = `rotate(${swing * 0.6}deg)`;
    legLeft.style.transformOrigin = "42.5px 80px";
    legRight.style.transformOrigin = "57.5px 80px";
    armLeft.style.transformOrigin = "22px 52px";
    armRight.style.transformOrigin = "78px 52px";
  }
}

renderBot();
