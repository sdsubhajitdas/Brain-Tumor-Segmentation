(() => {
  const uploadInput = document.getElementById("upload-input");
  const uploadSubmit = document.getElementById("upload-submit");
  const uploadFilename = document.getElementById("upload-filename");
  const sampleThumbs = document.querySelectorAll(".sample-thumb");

  const resultBox = document.getElementById("result-box");
  const resultStatus = document.getElementById("result-status");
  const resultInputPreview = document.getElementById("result-input-preview");
  const resultMask = document.getElementById("result-mask");
  const resultOverlay = document.getElementById("result-overlay");
  const resultMaskLoading = document.getElementById("result-mask-loading");
  const resultOverlayLoading = document.getElementById("result-overlay-loading");
  const resultDice = document.getElementById("result-dice");
  const resultTiming = document.getElementById("result-timing");

  let selectedFile = null;

  uploadInput.addEventListener("change", () => {
    selectedFile = uploadInput.files[0] || null;
    uploadSubmit.disabled = !selectedFile;
    uploadFilename.textContent = selectedFile ? selectedFile.name : "Choose an MRI slice (PNG/JPG)";
    sampleThumbs.forEach((btn) => btn.classList.remove("selected"));
  });

  uploadSubmit.addEventListener("click", () => {
    if (!selectedFile) return;
    const reader = new FileReader();
    reader.onload = () => {
      resultInputPreview.src = reader.result;
    };
    reader.readAsDataURL(selectedFile);

    const formData = new FormData();
    formData.append("file", selectedFile);
    runPredict(formData, { source: "upload", file_type: selectedFile.type });
  });

  sampleThumbs.forEach((btn) => {
    btn.addEventListener("click", () => {
      sampleThumbs.forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      selectedFile = null;
      uploadInput.value = "";
      uploadSubmit.disabled = true;
      uploadFilename.textContent = "Choose an MRI slice (PNG/JPG)";

      const sampleId = btn.dataset.sampleId;
      resultInputPreview.src = `/static/samples/${sampleId}.png`;

      const formData = new FormData();
      formData.append("sample_id", sampleId);
      runPredict(formData, { source: "sample", sample_id: sampleId });
    });
  });

  async function runPredict(formData, meta) {
    track("predict-start", meta);
    resultBox.hidden = false;
    resultStatus.textContent = "Running model...";
    resultStatus.classList.remove("error");
    resultMask.removeAttribute("src");
    resultOverlay.removeAttribute("src");
    resultDice.textContent = "";
    resultTiming.textContent = "";
    setLoading(resultMaskLoading, "Running");
    setLoading(resultOverlayLoading, "Running");

    try {
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await response.json();

      if (!response.ok) {
        const error = new Error(data.detail || "Prediction failed.");
        error.status = response.status;
        throw error;
      }

      resultStatus.textContent = "Done.";
      resultMask.src = `data:image/png;base64,${data.mask_png_base64}`;
      resultOverlay.src = `data:image/png;base64,${data.overlay_png_base64}`;
      setLoading(resultMaskLoading, null);
      setLoading(resultOverlayLoading, null);
      resultDice.textContent =
        data.dice_score !== null
          ? `Dice score vs. ground truth: ${data.dice_score.toFixed(4)}`
          : "No ground-truth mask available for uploaded images.";
      resultTiming.textContent = `Inference time: ${Math.round(data.inference_ms)}ms`;
      track("predict-success", {
        source: meta.source,
        inference_ms: Math.round(data.inference_ms),
        ...(data.dice_score !== null && { dice: Number(data.dice_score.toFixed(4)) }),
      });
    } catch (err) {
      track("predict-error", { source: meta.source, status: err.status ?? 0 });
      resultStatus.textContent = `Error: ${err.message}`;
      resultStatus.classList.add("error");
      setLoading(resultMaskLoading, "No series loaded");
      setLoading(resultOverlayLoading, "No series loaded");
    }
  }

  function setLoading(el, label) {
    if (!label) {
      el.style.display = "none";
      return;
    }
    el.style.display = "flex";
    el.textContent = "";
    const span = document.createElement("span");
    span.className = "font-mono text-[10px] text-mute tracking-widest uppercase" + (label === "Running" ? " animate-pulse" : "");
    span.textContent = label;
    el.appendChild(span);
  }
})();
