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
    if (data.status == 'success') {
      document.getElementById('result').innerText = `${name} registered!`;
      document.getElementById('face').src = data.face;
      document.getElementById('face').style.display = 'inline';
    } else {
      document.getElementById('result').innerText = 'Failed: ' + data.reason;
    }
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
    document.getElementById('face').src = data.face;
    document.getElementById('face').style.display = 'inline';
  });
}