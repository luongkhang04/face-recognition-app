const video = document.getElementById('video');
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

function getBase64Frame() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg');
}

function register() {
  const name = document.getElementById('name').value;
  const image = getBase64Frame();
  fetch('/save_face', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `name=${encodeURIComponent(name)}&image=${encodeURIComponent(image)}`
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById('result').innerText = data.status === 'success' ? 'Registered!' : 'Failed: ' + data.reason;
  });
}

function recognize() {
  const image = getBase64Frame();
  fetch('/recognize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `image=${encodeURIComponent(image)}`
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById('result').innerText = `Name: ${data.name}, Similarity: ${data.similarity.toFixed(2)}`;
  });
}