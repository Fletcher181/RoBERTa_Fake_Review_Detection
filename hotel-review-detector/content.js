console.log("CONTENT SCRIPT LOADED");

let lastCall = 0;

document.addEventListener("mouseup", () => {

    setTimeout(async () => {

        const now = Date.now();
        if (now - lastCall < 2000) return;

        const text = window.getSelection().toString().trim();

        if (!text || text.length < 20) return;

        lastCall = now;

        console.log("Sending:", text);

        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        const data = await res.json();

        showResult(data.label, data.confidence, text);

    }, 300);
});

function showResult(label, confidence, text) {

    const old = document.getElementById("ai-result");
    if (old) old.remove();

    const percent = (confidence * 100).toFixed(2);

    const result = document.createElement("div");
    result.id = "ai-result";

    result.innerHTML = `
        <div style="font-size:16px; font-weight:700; letter-spacing:0.3px;">
            ${label}
        </div>
        <div style="font-size:13px; margin-top:4px; opacity:0.95;">
            Confidence: ${percent}%
        </div>
    `;

    // POSITION
    result.style.position = "absolute";
    result.style.zIndex = "999999";

    // SIZE (bigger + cleaner)
    result.style.minWidth = "180px";
    result.style.padding = "12px 16px";
    result.style.borderRadius = "14px";

    // FONT (modern feel)
    result.style.fontFamily = `"Segoe UI", "Inter", "Arial", sans-serif`;

    // TEXT ALIGN
    result.style.textAlign = "center";

    // SOFT SHADOW (modern card look)
    result.style.boxShadow = "0 10px 25px rgba(0,0,0,0.18)";

    // BORDER (subtle)
    result.style.border = "1px solid rgba(255,255,255,0.2)";

    // COLOR + GRADIENT (this is the “nice” part)
    const intensity = Math.min(1, Math.max(0.5, confidence));

    if (label === "Deceptive") {
        result.style.background = `linear-gradient(135deg, rgba(231,76,60,${intensity}) 0%, rgba(192,57,43,${intensity}) 100%)`;
        result.style.color = "white";
    } else {
        result.style.background = `linear-gradient(135deg, rgba(46,204,113,${intensity}) 0%, rgba(39,174,96,${intensity}) 100%)`;
        result.style.color = "white";
    }

    // START ANIMATION STATE
    result.style.opacity = "0";
    result.style.transform = "translateY(12px) scale(0.98)";
    result.style.transition = "all 0.25s ease";

    document.body.appendChild(result);

    const selection = window.getSelection();
    if (!selection.rangeCount) return;

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    
    // EXTRA OFFSET (key fix)
    const offsetY = 70; // pushes box higher so it won't cover text
    
    const centerX = rect.left + rect.width / 2;
    
    // final position
    result.style.top = `${window.scrollY + rect.top - offsetY}px`;
    result.style.left = `${window.scrollX + centerX - result.offsetWidth / 2}px`;

    // TRIGGER ANIMATION
    requestAnimationFrame(() => {
        result.style.opacity = "1";
        result.style.transform = "translateY(0) scale(1)";
    });

    // AUTO REMOVE
    setTimeout(() => {
        result.style.opacity = "0";
        result.style.transform = "translateY(-10px) scale(0.98)";
        setTimeout(() => result.remove(), 300);
    }, 3000);
}