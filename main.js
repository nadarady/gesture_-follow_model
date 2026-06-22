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

let hands = null;          // the legacy Hands instance
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
    setStatus("ready", "model ready — drop a video");
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
  // kick off ffmpeg rotation fix — this is async, so we show a spinner
  // and disable everything until it finishes
  dropzone.classList.add("hidden");
  playBtn.disabled = true;
  resetBtn.disabled = true;
  setStatus("", "fixing video orientation…");
  fixVideoRotation(file);
}

// loads ffmpeg.wasm if not already loaded, then processes the video to
// physically bake any rotation metadata into the actual pixel data.
// why: canvas drawImage() doesn't consistently respect the display matrix
// rotation flag that phone cameras embed — confirmed cross-browser bug,
// no reliable JS workaround. ffmpeg.wasm is the only deterministic fix.
async function fixVideoRotation(file) {
  showProcessing("Fixing video orientation…", "Loading ffmpeg (first time only, ~6MB)");

  try {
    if (!ffmpegInstance) {
      const { FFmpeg } = FFmpegWASM;
      const { toBlobURL } = FFmpegUtil;
      ffmpegInstance = new FFmpeg();

      // core@0.12.6 is the most stable single-thread build — no SharedArrayBuffer
      // needed, works without special COOP/COEP headers for the core itself
      // (we still need COOP/COEP for the Worker that ffmpeg.wasm spawns, hence
      // the _headers file). using toBlobURL wraps each CDN asset in a same-origin
      // blob: URL, which sidesteps the cross-origin Worker path resolution bug
      // that affects the UMD build when loaded from CDN directly.
      const baseURL = "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/umd";
      await ffmpegInstance.load({
        coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, "text/javascript"),
        wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, "application/wasm"),
      });
    }

    showProcessing("Fixing video orientation…", "Processing…");

    // write the uploaded file into ffmpeg's virtual filesystem
    const inputName = "input" + file.name.slice(file.name.lastIndexOf("."));
    const inputData = new Uint8Array(await file.arrayBuffer());
    await ffmpegInstance.writeFile(inputName, inputData);

    // transpose=1 rotates 90° counter-clockwise to correct a -90° display
    // matrix. -display_rotation 0 tells ffmpeg to READ the source as if its
    // display matrix were 0 (i.e. not auto-apply it before our filter runs,
    // which would cause a double-rotation). the output has no rotation flag.
    // we verified this exact command on the user's actual phone video file.
    await ffmpegInstance.exec([
      "-display_rotation", "0",
      "-i", inputName,
      "-vf", "transpose=1",
      "-c:a", "copy",
      "output.mp4"
    ]);

    // read the fixed video back out and hand it to the browser
    const outputData = await ffmpegInstance.readFile("output.mp4");
    const outputBlob = new Blob([outputData.buffer], { type: "video/mp4" });
    const url = URL.createObjectURL(outputBlob);

    // clean up ffmpeg's virtual filesystem for next time
    await ffmpegInstance.deleteFile(inputName);
    await ffmpegInstance.deleteFile("output.mp4");

    hideProcessing();
    setVideoSource(url);

  } catch (err) {
    console.error("ffmpeg rotation fix failed:", err);
    // graceful fallback: if ffmpeg fails for any reason, load the original
    // file unchanged. better than showing nothing at all.
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

function showProcessing(title, sub) {
  procTitle.textContent = title;
  procSub.textContent = sub;
  processingEl.classList.add("visible");
}

function hideProcessing() {
  processingEl.classList.remove("visible");
}

window.addEventListener("resize", () => {
  if (video.videoWidth) {
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
  }
});

// ---- playback controls ----
//
// playing at normal (1x) speed while doing two detection passes a frame
// (the crop-and-zoom thing) means processing can fall behind the actual
// video clock — frames just get silently skipped when that happens, which
// looks exactly like "randomly doesn't see my hand." slowing playback down
// gives detection enough headroom to actually keep up with every frame
// instead of dropping whichever ones it didn't finish in time.
const PLAYBACK_RATE = 0.4;

playBtn.addEventListener("click", () => {
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
//
// this works differently than it used to, because the legacy Hands library
// is callback-based instead of synchronous. the flow per frame is now:
//   1. predictLoop fires on every rAF tick. if no detection is currently
//      in flight, kick one off (hands.send) and remember which crop rect
//      it was for.
//   2. sometime later (could be a few ms, could be most of a frame), the
//      result arrives via onHandsResults. that's where we actually update
//      the on-screen skeleton, recompute the hand's bounding box, and feed
//      the angle into the smoothing buffer.
// the crop-and-zoom idea (detect on a tight, upscaled crop instead of the
// whole frame, because mediapipe's hand detector struggles when the hand
// is small/off-center/angled) is unchanged — it's just wired up around an
// async call now instead of a synchronous one.
function predictLoop() {
  if (video.paused || video.ended) return;

  if (hands && !detectionInFlight && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    sendFrameForDetection();
  }

  stepBot();
  rafId = requestAnimationFrame(predictLoop);
}

// how often (in frames) to force a full-frame scan even if we currently
// have a crop fix — catches the hand if it jumps somewhere the crop
// doesn't cover, e.g. a cut/jump in the source footage
const FULL_SCAN_INTERVAL = 20;

function sendFrameForDetection() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;

  framesSinceFullScan++;
  const needsFullScan = !lastHandBox || framesSinceFullScan >= FULL_SCAN_INTERVAL;

  detectionInFlight = true;

  if (needsFullScan) {
    framesSinceFullScan = 0;
    pendingCropForInFlightDetection = null; // null = this result is in full-frame coordinates
    hands.send({ image: video });
    return;
  }

  // crop tight around where we last saw the hand, upscale it, and detect
  // on THAT instead of the full frame
  const crop = computeCropRect(lastHandBox, vw, vh);
  cropCanvas.width = CROP_SIZE;
  cropCanvas.height = CROP_SIZE;
  cropCtx.drawImage(
    video,
    crop.x, crop.y, crop.size, crop.size,  // source rect, in real video pixels
    0, 0, CROP_SIZE, CROP_SIZE             // dest rect, filling the upscaled canvas
  );

  pendingCropForInFlightDetection = crop;
  hands.send({ image: cropCanvas });
}

// the callback the legacy library calls once detection actually finishes.
// `results.multiHandLandmarks` is this library's name for what the newer
// API called `result.landmarks` — same 21-point shape per hand, different
// field name.
function onHandsResults(results) {
  const crop = pendingCropForInFlightDetection;
  detectionInFlight = false;

  const landmarksList = results.multiHandLandmarks;

  if (landmarksList && landmarksList.length > 0) {
    lastHandBox = boundingBoxFromLandmarks(landmarksList[0], crop, video.videoWidth, video.videoHeight);
  } else if (crop) {
    // missed inside a crop — don't immediately abandon the crop fix, the
    // presence-hysteresis logic in updateDirection handles brief gaps. but
    // if we keep missing, force a full rescan soon instead of staying
    // stuck staring at empty space.
    framesSinceFullScan = FULL_SCAN_INTERVAL - 3;
  } else {
    lastHandBox = null;
  }

  drawLandmarks(landmarksList, crop);
  updateDirection(landmarksList, crop);
}

// works out a square crop region (in real video pixel coordinates) centered
// on the last known hand position, padded generously so fast hand movement
// between frames doesn't immediately fall outside the crop
function computeCropRect(box, vw, vh) {
  if (!box) {
    // no fix yet — crop is just the whole frame
    const size = Math.min(vw, vh);
    return { x: (vw - size) / 2, y: (vh - size) / 2, size };
  }

  const PADDING_FACTOR = 2.6; // how much bigger than the hand's own bbox to crop
  const cx = box.cx * vw;
  const cy = box.cy * vh;
  let size = box.size * Math.max(vw, vh) * PADDING_FACTOR;

  // floor/ceiling so the crop never gets absurdly tiny (jittery zoom) or
  // bigger than the frame itself
  size = Math.max(size, Math.min(vw, vh) * 0.25);
  size = Math.min(size, Math.min(vw, vh));

  let x = cx - size / 2;
  let y = cy - size / 2;
  // clamp so the crop rectangle stays inside the actual video frame
  x = Math.max(0, Math.min(vw - size, x));
  y = Math.max(0, Math.min(vh - size, y));

  return { x, y, size };
}

// derives a rough center+size bounding box from a hand's 21 landmarks.
// if `crop`/`vw`/`vh` are provided, the landmarks are assumed to be in
// crop-normalized space and get remapped back to full-video-normalized
// space first.
function boundingBoxFromLandmarks(landmarks, crop, vw, vh) {
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of landmarks) {
    let x = p.x, y = p.y;
    if (crop) {
      // p.x/p.y are normalized within the crop (0..1) — convert to real
      // pixels within the crop, then to real video pixels, then back to
      // normalized full-video coordinates
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

// ---- draw the skeleton overlay on top of the video ----
//
// IMPORTANT MIRROR NOTE: mediapipe hands back landmark x/y normalized to the
// raw decoded video frame. that's NOT always the same orientation you see
// on screen — plenty of phone-recorded "selfie" videos store frames
// un-mirrored even though the phone's live preview mirrored them while you
// were filming. there's no universal flag for this, so rather than guess,
// we flip x consistently everywhere (drawing + direction math) to match
// what's visually on screen. if your hand ever again looks correct in the
// video but the dots float to the wrong side, this is the line to revisit.
const MIRROR_X = true;

// remaps a single landmark point from crop-normalized space (0..1 within
// the cropped/upscaled detection canvas) back to full-video-normalized
// space (0..1 within the original video frame). pass `null` for crop when
// the result already came from a full-frame detection pass — then it's a
// no-op passthrough.
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
  const vw = video.videoWidth, vh = video.videoHeight;

  for (const rawLandmarks of landmarksList) {
    // remap every point back to full-video space up front, so the rest of
    // this function doesn't need to care whether we're looking at a
    // full-frame result or a cropped-and-upscaled one
    const landmarks = crop ? rawLandmarks.map(p => remapPoint(p, crop, vw, vh)) : rawLandmarks;

    ctx.strokeStyle = "rgba(93,255,196,0.85)";
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      const p1 = landmarks[a], p2 = landmarks[b];
      const x1 = MIRROR_X ? (1 - p1.x) * w : p1.x * w;
      const x2 = MIRROR_X ? (1 - p2.x) * w : p2.x * w;
      ctx.beginPath();
      ctx.moveTo(x1, p1.y * h);
      ctx.lineTo(x2, p2.y * h);
      ctx.stroke();
    }
    landmarks.forEach((p, i) => {
      const px = MIRROR_X ? (1 - p.x) * w : p.x * w;
      ctx.beginPath();
      ctx.arc(px, p.y * h, i === 8 ? 5 : 3.2, 0, Math.PI * 2);
      ctx.fillStyle = i === 8 ? "#ffffff" : "#5dffc4"; // fingertip gets star treatment
      ctx.fill();
      if (i === 8) {
        ctx.strokeStyle = "rgba(93,255,196,0.9)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
  }
}

// ---- figure out which way the finger is pointing ----
// using the vector from the index MCP joint (5) to the fingertip (8) instead of
// wrist-to-tip. wrist-to-tip sounds more intuitive but it drifts whenever the
// whole hand rotates, even if the finger itself didn't change angle. MCP-to-tip
// only cares about the finger's own pose, which turned out to be way steadier.
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
    // if we're still inside the "off threshold" grace window, just hold
    // the last known heading rather than reacting to this one missed frame
    return;
  }

  consecutiveMisses = 0;
  consecutiveHits++;

  const vw = video.videoWidth, vh = video.videoHeight;
  const rawLm = landmarksList[0];
  const base = remapPoint(rawLm[5], crop, vw, vh);
  const tip = remapPoint(rawLm[8], crop, vw, vh);

  // same MIRROR_X flip as the drawing code, applied to x before computing
  // the vector — keeps the heading agreeing with what the skeleton shows
  // on screen, instead of the raw (possibly mirrored) frame data.
  const baseX = MIRROR_X ? 1 - base.x : base.x;
  const tipX = MIRROR_X ? 1 - tip.x : tip.x;

  const dx = tipX - baseX;
  const dy = tip.y - base.y;

  const mag = Math.hypot(dx, dy) || 1;
  // push this frame's reading into the buffer as a unit vector, not a raw
  // angle — see the big comment up top for why
  angleBuffer.push({ x: dx / mag, y: dy / mag });
  if (angleBuffer.length > SMOOTH_WINDOW) angleBuffer.shift();

  // average the buffered vectors, then re-derive the angle. this is the
  // standard trick for averaging angles without wraparound blowing up.
  let sumX = 0, sumY = 0;
  for (const v of angleBuffer) { sumX += v.x; sumY += v.y; }
  const avgX = sumX / angleBuffer.length;
  const avgY = sumY / angleBuffer.length;

  const angleRad = Math.atan2(avgY, avgX);
  const angleDeg = angleRad * (180 / Math.PI);

  // require a couple consistent hits before we actually trust this and
  // start moving — kills the "twitches the instant it glimpses something
  // hand-shaped for one frame" issue
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

  // shortest-path angle smoothing. without the wraparound correction here,
  // going from like 179° to -179° makes the bot do a dramatic 358° pirouette
  // instead of just nudging 2° further. learned that one the hard way.
  let diff = targetAngleDeg - botAngleDeg;
  diff = ((diff + 180) % 360 + 360) % 360 - 180;
  botAngleDeg += diff * 0.12;

  const rad = botAngleDeg * (Math.PI / 180);
  const worldRect = world.getBoundingClientRect();
  const halfW = worldRect.width / 2;
  const halfH = worldRect.height / 2;

  let nx = botPos.x + Math.cos(rad) * BOT_SPEED;
  let ny = botPos.y + Math.sin(rad) * BOT_SPEED;

  // pac-man rules: walk off one edge, show up on the other.
  //
  // got this backwards on the first attempt — the world panel has
  // overflow:hidden, so the bot needs to wrap the INSTANT its leading edge
  // would touch the panel boundary, not after its center clears some buffer
  // past it. otherwise it visually vanishes into the clipped edge for a
  // stretch before the wrap kicks in, which looks exactly like hitting a
  // wall. threshold is world-half-size minus the bot's own half-size.
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
  // bot's resting art faces "up" the screen, which is -90deg in atan2-land,
  // so we add 90deg to line its nose up with wherever it's actually heading.
  const visualOffset = 90;
  svg.style.transform =
    `translate(-50%, -50%) translate(${botPos.x}px, ${botPos.y}px) rotate(${botAngleDeg + visualOffset}deg)`;

  // tiny walk cycle — legs and arms swing opposite each other. purely
  // decorative, the direction math couldn't care less about this part.
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

// draw it once at rest so the page doesn't look broken before a video loads
renderBot();
