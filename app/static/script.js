// app/static/script.js
const startBtn = document.getElementById('startBtn');
const stopBtn  = document.getElementById('stopBtn');
const video    = document.getElementById('video');
const canvas   = document.getElementById('canvas');
const labelEl  = document.getElementById('label');
const confEl   = document.getElementById('conf');
const intervalInput = document.getElementById('interval');

let stream = null;
let captureInterval = null;
const ctx = canvas.getContext('2d');

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  if (stream) {
    for (const t of stream.getTracks()) t.stop();
    stream = null;
  }
}

async function sendFrame() {
  // draw current video frame to canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  // compress to jpeg dataURL
  const dataUrl = canvas.toDataURL('image/jpeg', 0.7); // reduce size
  try {
    const res = await fetch('/predict_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    });
    const j = await res.json();
    if (j.label) {
      labelEl.textContent = j.label;
      confEl.textContent = (j.confidence || 0).toFixed(2);
    } else if (j.error) {
      labelEl.textContent = "Error";
      confEl.textContent = j.error;
    }
  } catch (err) {
    console.error('Request failed', err);
  }
}

startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  stopBtn.disabled = false;
  await startCamera();
  const ms = parseInt(intervalInput.value) || 400;
  captureInterval = setInterval(sendFrame, ms);
});

stopBtn.addEventListener('click', () => {
  startBtn.disabled = false;
  stopBtn.disabled = true;
  if (captureInterval) clearInterval(captureInterval);
  stopCamera();
});
