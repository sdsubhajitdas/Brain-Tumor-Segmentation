(() => {
  const uploadInput = document.getElementById("upload-input");
  const uploadSubmit = document.getElementById("upload-submit");
  const sampleThumbs = document.querySelectorAll(".sample-thumb");

  const resultBox = document.getElementById("result-box");
  const resultStatus = document.getElementById("result-status");
  const resultInputPreview = document.getElementById("result-input-preview");
  const resultMask = document.getElementById("result-mask");
  const resultOverlay = document.getElementById("result-overlay");
  const resultDice = document.getElementById("result-dice");
  const resultTiming = document.getElementById("result-timing");

  let selectedFile = null;

  uploadInput.addEventListener("change", () => {
    selectedFile = uploadInput.files[0] || null;
    uploadSubmit.disabled = !selectedFile;
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
    runPredict(formData);
  });

  sampleThumbs.forEach((btn) => {
    btn.addEventListener("click", () => {
      sampleThumbs.forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      selectedFile = null;
      uploadInput.value = "";
      uploadSubmit.disabled = true;

      const sampleId = btn.dataset.sampleId;
      resultInputPreview.src = `/static/samples/${sampleId}.png`;

      const formData = new FormData();
      formData.append("sample_id", sampleId);
      runPredict(formData);
    });
  });

  async function runPredict(formData) {
    resultBox.hidden = false;
    resultStatus.textContent = "Running model...";
    resultStatus.classList.remove("error");
    resultMask.removeAttribute("src");
    resultOverlay.removeAttribute("src");
    resultDice.textContent = "";
    resultTiming.textContent = "";

    try {
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      resultStatus.textContent = "Done.";
      resultMask.src = `data:image/png;base64,${data.mask_png_base64}`;
      resultOverlay.src = `data:image/png;base64,${data.overlay_png_base64}`;
      resultDice.textContent =
        data.dice_score !== null
          ? `Dice score vs. ground truth: ${data.dice_score.toFixed(4)}`
          : "No ground-truth mask available for uploaded images.";
      resultTiming.textContent = `Inference time: ${Math.round(data.inference_ms)}ms`;
    } catch (err) {
      resultStatus.textContent = `Error: ${err.message}`;
      resultStatus.classList.add("error");
    }
  }
})();
