const video = document.getElementById("video");

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(startWebcam);

function startWebcam() {
  navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((error) => {
      console.error(error);
    });
}

video.addEventListener("play", async () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  faceapi.matchDimensions(canvas, { height: video.height, width: video.width });
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  console.log("facematcher", faceMatcher);
  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, {
      height: video.height,
      width: video.width,
    });

    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    //faceapi.draw.drawDetections(canvas, resizedDetections);
    //faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    const results = resizedDetections.map((d) =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    if (results[0]) {
      console.log("result: ", results[0], results[0].label);
      document.getElementById("name").innerText = results[0].label;
    } else {
      document.getElementById("name").innerText = "";
    }
    //console.log("detections: ", detections);
    await detectBlink(detections);
  }, 100);
});

function loadLabeledImages() {
  const labels = [
    {
      name: "Subrata Kumar Das",
      url: "./faces/Subrata Kumar Das/1.jpg",
    },
  ];
  return Promise.all(
    labels.map(async ({ name, url }) => {
      const descriptions = [];
      const img = await faceapi.fetchImage(url);
      const detections = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();
      descriptions.push(detections.descriptor);

      return new faceapi.LabeledFaceDescriptors(name, descriptions);
    })
  );
}

async function detectBlink(detections) {
  detections.forEach((detection) => {
    const landmarks = detection.landmarks;
    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();

    const leftEAR = calculateEAR(leftEye);
    const rightEAR = calculateEAR(rightEye);
    const ear = (leftEAR + rightEAR) / 2;

    const BLINK_THRESHOLD = 0.25;
    if (ear < BLINK_THRESHOLD) {
      console.log("Blink detected");
      document.getElementById("liveness").innerText = "Liveness: Live";
    } else {
      document.getElementById("liveness").innerText = "Liveness: Not Live";
    }
  });
}

function calculateEAR(eye) {
  const A = Math.sqrt(
    Math.pow(eye[1].x - eye[5].x, 2) + Math.pow(eye[1].y - eye[5].y, 2)
  );
  const B = Math.sqrt(
    Math.pow(eye[2].x - eye[4].x, 2) + Math.pow(eye[2].y - eye[4].y, 2)
  );
  const C = Math.sqrt(
    Math.pow(eye[0].x - eye[3].x, 2) + Math.pow(eye[0].y - eye[3].y, 2)
  );
  const ear = (A + B) / (2.0 * C);
  return ear;
}
