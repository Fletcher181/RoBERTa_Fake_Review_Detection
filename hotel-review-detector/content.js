console.log("CONTENT SCRIPT LOADED");
console.log("Highlight event triggered");
let lastCall = 0;

document.addEventListener("mouseup", () => {

    console.log("Mouseup detected");

    setTimeout(async () => {

        const now = Date.now();
        if (now - lastCall < 2000) return;

        const text = window.getSelection().toString().trim();

        console.log("Selected text:", text);

        if (!text || text.length < 20) return;

        lastCall = now;

        console.log("Calling API...");

        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        const data = await res.json();

        console.log("API response:", data);

        showResult(data.label, data.confidence, text);

    }, 300);
});

function showResult(label, confidence, text) {

    // remove old popup
    const old = document.getElementById("ai-popup");
    if (old) old.remove();

    // get selection position
    const selection = window.getSelection();
    if (!selection.rangeCount) return;

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();

    // create popup
    const box = document.createElement("div");
    box.id = "ai-popup";

    // IMPORTANT: absolute positioning relative to page
    box.style.position = "absolute";

    // ✅ CENTER ABOVE HIGHLIGHT
    const popupWidth = 320;

    box.style.left = `${window.scrollX + rect.left + rect.width / 2 - popupWidth / 2}px`;
    box.style.top = `${window.scrollY + rect.top}px`;
    box.style.transform = "translateY(-110%)";

    box.style.zIndex = "999999";

    // styling
    box.style.width = `${popupWidth}px`;
    box.style.padding = "18px";
    box.style.borderRadius = "14px";
    box.style.background = "white";
    box.style.boxShadow = "0 10px 25px rgba(0,0,0,0.2)";
    box.style.fontFamily = "Segoe UI, sans-serif";
    box.style.textAlign = "center";

    const color = label === "Deceptive" ? "#e74c3c" : "#2ecc71";

    box.innerHTML = `
        <div style="
            font-size:20px;
            font-weight:700;
            color:${color};
            margin-bottom:10px;
        ">
            ${label}
        </div>

        <div style="
            font-size:15px;
            margin-bottom:15px;
        ">
            Confidence: ${(confidence * 100).toFixed(2)}%
        </div>

        <button id="view-details-btn"
            style="
                padding:10px 14px;
                border:none;
                border-radius:8px;
                background:${color};
                color:white;
                font-weight:600;
                cursor:pointer;
                width:100%;
                font-size:14px;
            ">
            View Details
        </button>
    `;

    document.body.appendChild(box);

    // button action
    document.getElementById("view-details-btn")
        .addEventListener("click", () => {

            const url = `http://127.0.0.1:8000/dashboard?label=${label}&confidence=${(confidence * 100).toFixed(2)}`;

            window.open(url, "_blank");
        });
}