/* Browser camera helpers shared by the attendance and registration pages. */
(function () {
  "use strict";

  var MAX_SIDE = 800; // longest side (px) of frames sent to the server

  function sleep(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  function token() {
    var bytes = new Uint8Array(16);
    window.crypto.getRandomValues(bytes);
    return Array.prototype.map.call(bytes, function (b) {
      return ("0" + b.toString(16)).slice(-2);
    }).join("");
  }

  async function startCamera(video) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera access needs a secure page. Open this app as " +
        "http://localhost:PORT (or over HTTPS), not by IP address.");
    }
    var stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    if (!video.videoWidth) {
      await new Promise(function (resolve) { video.onloadedmetadata = resolve; });
    }
    return stream;
  }

  function stopCamera(stream, video) {
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
    }
    if (video) { video.srcObject = null; }
  }

  function describeError(e) {
    var name = e && e.name;
    if (name === "NotAllowedError" || name === "SecurityError") {
      return "Camera permission was denied. Allow camera access for this site " +
        "in Chrome (lock icon > Permissions) and try again.";
    }
    if (name === "NotFoundError" || name === "OverconstrainedError") {
      return "No camera was found on this device.";
    }
    if (name === "NotReadableError") {
      return "The camera is being used by another app. Close it and try again.";
    }
    return (e && e.message) || "Could not start the camera.";
  }

  function grabFrame(video, canvas) {
    var scale = Math.min(1, MAX_SIDE / Math.max(video.videoWidth, video.videoHeight));
    var w = Math.round(video.videoWidth * scale);
    var h = Math.round(video.videoHeight * scale);
    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d").drawImage(video, 0, 0, w, h);
    return new Promise(function (resolve, reject) {
      canvas.toBlob(function (blob) {
        if (blob) { resolve({ blob: blob, w: w, h: h }); }
        else { reject(new Error("Could not read a frame from the camera.")); }
      }, "image/jpeg", 0.85);
    });
  }

  function drawFace(canvas, frame, face, label, color) {
    if (canvas.width !== frame.w) { canvas.width = frame.w; }
    if (canvas.height !== frame.h) { canvas.height = frame.h; }
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!face) { return; }
    ctx.lineWidth = 4;
    ctx.strokeStyle = color;
    ctx.strokeRect(face.x, face.y, face.w, face.h);
    if (label) {
      ctx.font = "bold 28px sans-serif";
      ctx.textBaseline = "middle";
      var pad = 8;
      var width = ctx.measureText(label).width + pad * 2;
      var top = face.y > 44 ? face.y - 40 : face.y + face.h;
      ctx.fillStyle = color;
      ctx.fillRect(face.x, top, width, 36);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, face.x + pad, top + 18);
    }
  }

  async function parse(resp) {
    var data = {};
    try { data = await resp.json(); } catch (e) { /* non-JSON error page */ }
    if (!resp.ok) {
      throw new Error(data.error || "Server error (" + resp.status + ")");
    }
    return data;
  }

  async function postJSON(url, body) {
    var resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    return parse(resp);
  }

  async function postFrame(url, fields, blob) {
    var form = new FormData();
    Object.keys(fields).forEach(function (k) { form.append(k, fields[k]); });
    form.append("image", blob, "frame.jpg");
    var resp = await fetch(url, { method: "POST", body: form });
    return parse(resp);
  }

  window.Cam = {
    sleep: sleep, token: token, startCamera: startCamera, stopCamera: stopCamera,
    describeError: describeError, grabFrame: grabFrame, drawFace: drawFace,
    postJSON: postJSON, postFrame: postFrame
  };
})();
