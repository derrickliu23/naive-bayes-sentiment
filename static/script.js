// script.js
// Handles the classify button, calls the Flask API, and renders results.

const btn = document.getElementById("classify-btn");
const textarea = document.getElementById("input-text");
const resultBanner = document.getElementById("result-banner");
const resultLabel = document.getElementById("result-label");
const resultConfidence = document.getElementById("result-confidence");
const barNegative = document.getElementById("bar-negative");
const barPositive = document.getElementById("bar-positive");
const evidenceSection = document.getElementById("evidence-section");
const evidenceBody = document.getElementById("evidence-body");

// ── Main click handler ──────────────────────────────────────────────────────

btn.addEventListener("click", async () => {
  const text = textarea.value.trim();
  if (!text) return;

  // Disable button while request is in flight
  btn.disabled = true;
  btn.textContent = "Classifying...";

  try {
    const response = await fetch("/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    renderResult(data);
  } catch (err) {
    alert("Something went wrong: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Classify";
  }
});

// Allow Ctrl+Enter / Cmd+Enter to submit
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    btn.click();
  }
});

// ── Render helpers ──────────────────────────────────────────────────────────

function renderResult(data) {
  const { label, confidence, probabilities, explanation } = data;

  // -- Result banner --
  resultLabel.textContent = label;
  resultConfidence.textContent =
    `Confidence: ${(confidence * 100).toFixed(1)}%`;

  // Swap color class based on predicted label
  resultBanner.classList.remove("is-positive", "is-negative");
  resultBanner.classList.add(label === "positive" ? "is-positive" : "is-negative");

  // Probability bar: split the track into negative | positive proportions
  const negPct = ((probabilities.negative ?? 0) * 100).toFixed(1);
  const posPct = ((probabilities.positive ?? 0) * 100).toFixed(1);
  barNegative.style.width = negPct + "%";
  barPositive.style.width = posPct + "%";

  // Show the banner
  resultBanner.classList.remove("hidden");

  // -- Evidence table --
  renderEvidence(explanation);
  evidenceSection.classList.remove("hidden");
}

function renderEvidence(explanation) {
  // Find the largest absolute diff so we can scale influence bars relatively
  const maxAbs = Math.max(...explanation.map((e) => Math.abs(e.diff)));

  evidenceBody.innerHTML = explanation
    .map((entry) => {
      const { word, diff, scores } = entry;
      const isPos = diff >= 0;
      const diffClass = isPos ? "diff-pos" : "diff-neg";
      const diffSign = isPos ? "+" : "";
      const barColor = isPos ? "#4ade80" : "#f87171";

      // Scale bar width to max influence seen in this result set
      const barWidth = maxAbs > 0
        ? ((Math.abs(diff) / maxAbs) * 100).toFixed(1)
        : 0;

      // Influence bar: fills left-to-right for positive, right-to-left for negative
      const barStyle = isPos
        ? `left: 0; width: ${barWidth}%; background: ${barColor};`
        : `right: 0; width: ${barWidth}%; background: ${barColor};`;

      return `
        <tr>
          <td class="word">${escapeHtml(word)}</td>
          <td>
            <div class="influence-bar-track">
              <div class="influence-bar-fill" style="${barStyle}"></div>
            </div>
          </td>
          <td>${scores.positive ?? "—"}</td>
          <td>${scores.negative ?? "—"}</td>
          <td class="${diffClass}">${diffSign}${diff.toFixed(4)}</td>
        </tr>
      `;
    })
    .join("");
}

// Prevent XSS from user-typed text being injected into the table
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}