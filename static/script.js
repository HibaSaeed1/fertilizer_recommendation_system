(() => {
  "use strict";

  const form = document.getElementById("fert-form");
  const errorBox = document.getElementById("error");
  const submitBtn = document.getElementById("submit-btn");
  const btnText = document.querySelector(".btn-text");
  const spinner = document.getElementById("spinner");

  const result = document.getElementById("result");
  const resultName = document.getElementById("result-name");
  const matchBadge = document.getElementById("match-badge");
  const confidenceFill = document.getElementById("confidence-fill");
  const confidencePct = document.getElementById("confidence-pct");
  const altWrap = document.getElementById("alt-wrap");
  const altList = document.getElementById("alt-list");

  const soilSelect = document.getElementById("Soil Type");
  const cropSelect = document.getElementById("Crop Type");

  async function loadOptions() {
    try {
      const res = await fetch("/options");
      if (!res.ok) throw new Error();
      const data = await res.json();
      fillSelect(soilSelect, data.soil_types);
      fillSelect(cropSelect, data.crop_types);
    } catch {
      showError("Could not load soil and crop options. Confirm the server is running, then reload the page.");
    }
  }

  function fillSelect(select, values) {
    select.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.disabled = true;
    placeholder.selected = true;
    placeholder.value = "";
    placeholder.textContent = "Select…";
    select.appendChild(placeholder);

    values.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      select.appendChild(opt);
    });
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function hideError() {
    errorBox.classList.add("hidden");
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    btnText.textContent = isLoading ? "Analyzing…" : "Recommend Fertilizer";
    spinner.classList.toggle("hidden", !isLoading);
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();
    setLoading(true);
    result.classList.add("hidden");
    confidenceFill.style.width = "0%";

    const payload = Object.fromEntries(new FormData(form).entries());

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong. Please try again.");

      resultName.textContent = data.recommended_fertilizer;

      if (data.exact_match) {
        matchBadge.textContent = "Matched known sample";
        matchBadge.className = "match-badge known";
      } else {
        matchBadge.textContent = "Model prediction";
        matchBadge.className = "match-badge model";
      }
      matchBadge.classList.remove("hidden");

      const top = (data.top3 || [])[0];
      const pct = top ? Math.max(0, Math.min(100, top.confidence)) : 0;
      confidencePct.textContent = top ? `${top.confidence.toFixed(1)}%` : "";

      const alts = (data.top3 || []).slice(1);
      altList.innerHTML = "";
      alts.forEach((item) => {
        const altPct = Math.max(0, Math.min(100, item.confidence));
        const li = document.createElement("li");
        li.innerHTML = `
          <span class="alt-name">${item.fertilizer}</span>
          <span class="alt-track"><span class="alt-fill" style="width:${altPct}%"></span></span>
          <span class="alt-pct">${item.confidence.toFixed(1)}%</span>`;
        altList.appendChild(li);
      });
      altWrap.classList.toggle("hidden", alts.length === 0);

      result.classList.remove("hidden");
      // animate the confidence bar in on the next frame, after layout settles
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          confidenceFill.style.width = `${pct}%`;
        });
      });

      result.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  });

  loadOptions();
})();