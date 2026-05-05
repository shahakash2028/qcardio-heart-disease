/**
 * script.js — Q-Cardio AI Frontend
 * Handles: custom dropdowns · gauge animation · loader steps
 *          · API call · result rendering · grid canvas
 *
 * Response format from updated app.py:
 *   { prediction, label, probability, status, raw_q_prob, stack_prob }
 */

const API_URL = "http://localhost:5000/predict";

// ── Feature IDs must match config.FEATURE_NAMES exactly ──────────────────
const FEATURE_IDS = [
    "age","sex","cp","trestbps","chol",
    "fbs","restecg","thalach","exang",
    "oldpeak","slope","ca","thal"
];

// SVG ring circumference = 2π × 65 ≈ 408.41
const GAUGE_CIRC = 408.41;

// ══════════════════════════════════════════════════════════════════════════
// 1. GRID CANVAS
// ══════════════════════════════════════════════════════════════════════════
(function initGrid() {
    const canvas = document.getElementById("gridCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    function resize() {
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
        draw();
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const step = 52;
        ctx.strokeStyle = "rgba(0,245,212,0.04)";
        ctx.lineWidth   = 1;
        for (let x = 0; x <= canvas.width; x += step) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
        }
        for (let y = 0; y <= canvas.height; y += step) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }
    }

    window.addEventListener("resize", resize);
    resize();
})();

// ══════════════════════════════════════════════════════════════════════════
// 2. CUSTOM SELECT DROPDOWNS
// ══════════════════════════════════════════════════════════════════════════
function initCustomSelects() {
    document.querySelectorAll(".custom-select").forEach(sel => {
        const display = sel.querySelector(".cs-display");
        const items   = sel.querySelectorAll(".cs-item");
        const text    = sel.querySelector(".cs-text");

        display.addEventListener("click", e => {
            e.stopPropagation();
            const isOpen = sel.classList.contains("open");
            document.querySelectorAll(".custom-select.open").forEach(s => s.classList.remove("open"));
            if (!isOpen) sel.classList.add("open");
        });

        items.forEach(item => {
            item.addEventListener("click", e => {
                e.stopPropagation();
                items.forEach(i => i.classList.remove("active"));
                item.classList.add("active");
                text.textContent  = item.textContent.trim();
                sel.dataset.value = item.dataset.value;
                sel.classList.remove("open");
                display.style.borderColor = "var(--teal)";
                setTimeout(() => { display.style.borderColor = ""; }, 600);
            });
        });
    });

    document.addEventListener("click", () => {
        document.querySelectorAll(".custom-select.open").forEach(s => s.classList.remove("open"));
    });
}

// ══════════════════════════════════════════════════════════════════════════
// 3. COLLECT FORM VALUES
// ══════════════════════════════════════════════════════════════════════════
function collectPayload() {
    const payload = {};
    FEATURE_IDS.forEach(id => {
        const sel = document.querySelector(`.custom-select[data-id="${id}"]`);
        if (sel) {
            payload[id] = parseFloat(sel.dataset.value) || 0;
        } else {
            const inp = document.getElementById(id);
            payload[id] = inp ? (parseFloat(inp.value) || 0) : 0;
        }
    });
    return payload;
}

// ══════════════════════════════════════════════════════════════════════════
// 4. UI STATE MANAGEMENT
// ══════════════════════════════════════════════════════════════════════════
const $idle    = document.getElementById("resultIdle");
const $loader  = document.getElementById("resultLoader");
const $content = document.getElementById("resultContent");
const $btn     = document.getElementById("analyzeBtn");
const $btnLabel= document.querySelector(".btn-label");

function showIdle() {
    $idle.classList.remove("hidden");
    $loader.classList.add("hidden");
    $content.classList.add("hidden");
}

function showLoader() {
    $idle.classList.add("hidden");
    $loader.classList.remove("hidden");
    $content.classList.add("hidden");
    $btn.disabled = true;
    $btnLabel.textContent = "Processing…";
    animateSteps();
}

function showContent() {
    $idle.classList.add("hidden");
    $loader.classList.add("hidden");
    $content.classList.remove("hidden");
    $btn.disabled = false;
    $btnLabel.textContent = "Run Quantum Analysis";
}

function animateSteps() {
    const steps = ["step1","step2","step3"];
    let i = 0;
    steps.forEach(s => document.getElementById(s).classList.remove("active"));
    document.getElementById(steps[0]).classList.add("active");
    const timer = setInterval(() => {
        i++;
        if (i >= steps.length) { clearInterval(timer); return; }
        steps.forEach(s => document.getElementById(s).classList.remove("active"));
        document.getElementById(steps[i]).classList.add("active");
    }, 900);
}

// ══════════════════════════════════════════════════════════════════════════
// 5. GAUGE ANIMATION
// ══════════════════════════════════════════════════════════════════════════
function animateGauge(prob) {
    const fill = document.getElementById("gaugeFill");
    const pct  = document.getElementById("gaugePct");

    const offset = GAUGE_CIRC * (1 - prob);
    fill.style.strokeDashoffset = offset;

    if (prob < 0.35) {
        fill.style.stroke = "#30e87a";
    } else if (prob < 0.60) {
        fill.style.stroke = "#f5a623";
    } else {
        fill.style.stroke = "url(#gaugeGrad)";
    }

    // Animate counter
    let current = 0;
    const target = Math.round(prob * 100);
    const step   = Math.max(1, Math.floor(target / 40));
    const timer  = setInterval(() => {
        current = Math.min(current + step, target);
        pct.textContent = current + "%";
        if (current >= target) clearInterval(timer);
    }, 25);
}

// ══════════════════════════════════════════════════════════════════════════
// 6. RENDER RESULT
// ══════════════════════════════════════════════════════════════════════════
const MESSAGES = {
    high: [
        "High cardiovascular risk detected. Consult a cardiologist immediately.",
        "Multiple risk indicators present. Clinical evaluation is strongly advised.",
        "Elevated risk factors found. Please seek specialist review promptly."
    ],
    mid: [
        "Moderate risk detected. Follow-up with your physician is recommended.",
        "Borderline indicators present. Medical review advised within 30 days.",
        "Intermediate risk profile. Lifestyle assessment with a doctor is advised."
    ],
    low: [
        "Low heart disease risk detected. Maintain a healthy active lifestyle.",
        "Favorable clinical indicators observed. Routine check-ups recommended.",
        "Results suggest low cardiac risk. Continue preventive care and monitoring."
    ]
};
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

function renderResult(data) {
    const prob       = data.probability  ?? 0;
    const prediction = data.prediction   ?? 0;
    const rawQ       = data.raw_q_prob   ?? prob;
    const stackP     = data.stack_prob   ?? prob;

    animateGauge(prob);

    // Verdict
    const verdict = document.getElementById("verdict");
    const icon    = document.getElementById("verdictIcon");
    const label   = document.getElementById("verdictLabel");

    verdict.classList.remove("positive","negative");
    if (prediction === 1) {
        verdict.classList.add("positive");
        icon.textContent  = "🫀";
        label.textContent = "Heart Disease Detected";
    } else {
        verdict.classList.add("negative");
        icon.textContent  = "💚";
        label.textContent = "No Heart Disease";
    }

    // Recommendation
    const msgGroup = prob > 0.65 ? "high" : prob > 0.40 ? "mid" : "low";
    document.getElementById("recText").textContent = pick(MESSAGES[msgGroup]);

    // Stats — show stack prob in the quantum slot if quantum not available
    const displayQ = (rawQ !== null && rawQ !== undefined) ? rawQ.toFixed(4) : stackP.toFixed(4);
    document.getElementById("rawQVal").textContent    = displayQ;
    document.getElementById("threshVal").textContent  = "0.45";
    document.getElementById("ensembleMode").textContent = "75/25";

    // Confidence bars
    setTimeout(() => {
        document.querySelectorAll(".conf-fill").forEach(bar => {
            bar.style.width = bar.dataset.w;
        });
    }, 400);
}

function renderError(msg) {
    const verdict = document.getElementById("verdict");
    verdict.classList.remove("positive","negative");
    document.getElementById("verdictIcon").textContent = "⚠️";
    document.getElementById("verdictLabel").textContent = "Error";
    document.getElementById("recText").textContent = msg || "Could not reach the Flask backend on port 5000.";
    document.getElementById("rawQVal").textContent = "–";
    animateGauge(0);
    setTimeout(() => {
        document.querySelectorAll(".conf-fill").forEach(bar => {
            bar.style.width = bar.dataset.w;
        });
    }, 400);
}

// ══════════════════════════════════════════════════════════════════════════
// 7. FORM SUBMIT → API CALL
// ══════════════════════════════════════════════════════════════════════════
document.getElementById("predictionForm").addEventListener("submit", async e => {
    e.preventDefault();
    showLoader();

    // Reset bars
    document.querySelectorAll(".conf-fill").forEach(b => { b.style.width = "0"; });

    const payload = collectPayload();
    console.log("📤 Sending payload:", payload);

    try {
        const res = await fetch(API_URL, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        console.log("📥 Response:", data);

        showContent();

        if (data.status === "fallback" || data.status === "error") {
            renderError(data.detail || "Backend error. Check server logs for details.");
        } else {
            renderResult(data);
        }

    } catch (err) {
        console.error("❌ Fetch error:", err);
        showContent();
        renderError("Connection failed. Is Flask running on http://localhost:5000?");
    }
});

// ══════════════════════════════════════════════════════════════════════════
// 8. INIT
// ══════════════════════════════════════════════════════════════════════════
initCustomSelects();
showIdle();