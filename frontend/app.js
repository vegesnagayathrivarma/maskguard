const API = "http://127.0.0.1:5000";

let stream = null;
let detectInterval = null;
let intervalMs = 200;
let frameCount = 0;
let lastFpsTime = Date.now();
let isProcessing = false;

const video  = document.getElementById("video");
const canvas = document.getElementById("canvas");
const output = document.getElementById("output");
const ctx    = canvas.getContext("2d");

// ── Start Camera ──
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false
    });

    video.srcObject = stream;

    // Wait for video to be ready
    video.onloadedmetadata = () => {
      video.play();
      document.getElementById("startBtn").disabled = true;
      document.getElementById("stopBtn").disabled = false;
      document.getElementById("noDetection").style.display = "none";
      hideError();

      // Start detection loop after video is playing
      setTimeout(() => {
        detectInterval = setInterval(sendFrame, intervalMs);
      }, 500);
    };

  } catch (err) {
    showError("Camera access denied. Please allow camera permission and refresh.");
    console.error(err);
  }
}

// ── Stop Camera ──
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  clearInterval(detectInterval);
  detectInterval = null;
  isProcessing = false;

  video.srcObject = null;
  output.src = "";

  document.getElementById("startBtn").disabled = false;
  document.getElementById("stopBtn").disabled = true;
  document.getElementById("noDetection").style.display = "block";
  document.getElementById("resultsPanel").classList.add("hidden");

  updateStats(0, 0, 0);
  document.getElementById("statFPS").textContent = "0";
}

// ── Capture & Send Frame ──
async function sendFrame() {
  if (!stream || isProcessing) return;
  if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

  isProcessing = true;

  try {
    // Draw current video frame to canvas
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to base64 JPEG
    const dataURL = canvas.toDataURL("image/jpeg", 0.85);
    const base64  = dataURL.replace("data:image/jpeg;base64,", "");

    const res = await fetch(`${API}/api/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 })
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    if (data.error) { showError(data.error); return; }

    // Show annotated image
    output.src = `data:image/jpeg;base64,${data.image}`;
    document.getElementById("noDetection").style.display = "none";

    // Update stats
    const withMask    = data.results.filter(r => r.label === "with_mask").length;
    const withoutMask = data.results.filter(r => r.label === "without_mask").length;
    updateStats(data.faces_detected, withMask, withoutMask);
    showResults(data.results);

    // FPS counter
    frameCount++;
    const now = Date.now();
    if (now - lastFpsTime >= 1000) {
      document.getElementById("statFPS").textContent = frameCount;
      frameCount = 0;
      lastFpsTime = now;
    }

  } catch (err) {
    console.log("Frame error:", err.message);
  } finally {
    isProcessing = false;
  }
}

// ── Update Stats ──
function updateStats(faces, withMask, withoutMask) {
  document.getElementById("statFaces").textContent  = faces;
  document.getElementById("statMask").textContent   = withMask;
  document.getElementById("statNoMask").textContent = withoutMask;
}

// ── Show Results ──
function showResults(results) {
  const panel = document.getElementById("resultsPanel");
  const list  = document.getElementById("resultsList");

  if (results.length === 0) {
    panel.classList.add("hidden");
    return;
  }

  panel.classList.remove("hidden");
  list.innerHTML = results.map((r, i) => {
    const isMask = r.label === "with_mask";
    const badge  = isMask ? "badge-mask" : "badge-nomask";
    const label  = isMask ? "✅ With Mask" : "❌ No Mask";
    const color  = isMask ? "#00e5aa" : "#ff5c5c";
    return `
      <div class="result-item">
        <span>Face ${i + 1}</span>
        <span class="badge ${badge}">${label}</span>
        <div class="confidence-bar">
          <div class="confidence-fill" style="width:${r.confidence}%;background:${color}"></div>
        </div>
        <span style="font-size:0.8rem;color:#888">${r.confidence}%</span>
      </div>`;
  }).join("");
}

// ── Update Detection Interval ──
function updateInterval() {
  intervalMs = parseInt(document.getElementById("intervalSelect").value);
  if (detectInterval) {
    clearInterval(detectInterval);
    detectInterval = setInterval(sendFrame, intervalMs);
  }
}

// ── Helpers ──
function showError(msg) {
  const el = document.getElementById("error");
  el.textContent = "❌ " + msg;
  el.classList.remove("hidden");
}

function hideError() {
  document.getElementById("error").classList.add("hidden");
}