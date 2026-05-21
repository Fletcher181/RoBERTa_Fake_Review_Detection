from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import torch
import torch.nn.functional as F
import numpy as np

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification
)

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="Fake Review Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# LOAD MODEL + TOKENIZER
# =========================================================

MODEL_PATH = "./roberta_fake_review_model"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model     = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
model.eval()

print(f"Model loaded on: {DEVICE}")

# =========================================================
# REQUEST SCHEMA
# =========================================================

class Review(BaseModel):
    text: str

# =========================================================
# ROOT ROUTE
# =========================================================

@app.get("/")
def home():
    return {"message": "Fake Review Detection API is running", "status": "ok"}

# =========================================================
# TOKEN ATTRIBUTION (Gradient x Input saliency)
# =========================================================

def get_token_attributions(text: str, pred_class: int):
    """
    Computes per-token importance using Gradient x Input saliency.
    Returns list of (word, score) pairs where score in [-1, 1].
    Positive score = pushed toward predicted class.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Get embeddings with gradient tracking
    embeddings = model.roberta.embeddings(inputs["input_ids"])
    embeddings.retain_grad()
    embeddings_input = embeddings.clone().detach().requires_grad_(True)

    # Forward pass through the rest of the model using embedding input
    outputs = model(
        inputs_embeds=embeddings_input,
        attention_mask=inputs["attention_mask"]
    )

    # Target the predicted class logit
    score = outputs.logits[0, pred_class]
    model.zero_grad()
    score.backward()

    # Gradient x Input saliency — shape: (seq_len, hidden)
    saliency = (embeddings_input.grad * embeddings_input).detach()

    # Sum across hidden dim → (seq_len,)
    token_scores = saliency.sum(dim=-1).squeeze(0).cpu().numpy()

    # Normalize to [-1, 1]
    max_abs = np.abs(token_scores).max()
    if max_abs > 0:
        token_scores = token_scores / max_abs

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"].squeeze(0).cpu().tolist()
    )

    # Merge subword tokens and skip special tokens
    words = []
    skip = {"<s>", "</s>", "<pad>"}

    for token, score in zip(tokens, token_scores):
        if token in skip:
            continue
        # RoBERTa uses Ġ prefix for word-start tokens
        if token.startswith("Ġ") or not words:
            words.append({
                "word":  token.replace("Ġ", ""),
                "score": float(score)
            })
        else:
            # Subword continuation — merge with previous word
            words[-1]["word"]  += token
            words[-1]["score"] += float(score)

    # Re-normalize after merging
    max_abs = max(abs(w["score"]) for w in words) if words else 1.0
    if max_abs > 0:
        for w in words:
            w["score"] = round(w["score"] / max_abs, 4)

    return words


# =========================================================
# PREDICT ROUTE
# =========================================================

@app.post("/predict")
def predict(review: Review):
    inputs = tokenizer(
        review.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs                = F.softmax(outputs.logits, dim=1)
    confidence, pred_cls = torch.max(probs, dim=1)
    pred_class           = int(pred_cls.item())

    label_map = {0: "Genuine", 1: "Deceptive"}
    label     = label_map[pred_class]

    # Compute token attributions
    token_scores = get_token_attributions(review.text, pred_class)

    return {
        "label":        label,
        "confidence":   float(confidence.item()),
        "token_scores": token_scores
    }


# =========================================================
# DASHBOARD ROUTE
# =========================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    label:        str = "",
    confidence:   str = "",
    text:         str = "",
    token_scores: str = ""
):
    import json, urllib.parse

    is_deceptive   = label == "Deceptive"
    verdict_color  = "#F0527A" if is_deceptive else "#52C9A0"
    verdict_bg     = "rgba(240,82,122,0.10)" if is_deceptive else "rgba(82,201,160,0.10)"
    verdict_icon   = "&#9888;" if is_deceptive else "&#10003;"
    conf_display   = f"{confidence}%" if confidence else "&#8212;"

    tokens_data = []
    if token_scores:
        try:
            tokens_data = json.loads(urllib.parse.unquote(token_scores))
        except Exception:
            tokens_data = []

    top_words = sorted(tokens_data, key=lambda x: abs(x["score"]), reverse=True)
    top_words = [w for w in top_words if w["word"].strip()][:8]

    def score_to_style(score):
        intensity = min(abs(score), 1.0)
        alpha     = 0.15 + intensity * 0.55
        if score > 0.1:
            color = "240,82,122" if is_deceptive else "82,201,160"
        elif score < -0.1:
            color = "82,201,160" if is_deceptive else "240,82,122"
        else:
            return ""
        return f"background:rgba({color},{alpha:.2f});border-radius:3px;padding:1px 3px;"

    highlighted_html = ""
    if tokens_data:
        for w in tokens_data:
            style = score_to_style(w["score"])
            word  = w["word"].replace("<","&lt;").replace(">","&gt;")
            if style:
                highlighted_html += f'<span style="{style}" title="Score: {w["score"]:.3f}">{word}</span> '
            else:
                highlighted_html += f'{word} '
    elif text:
        highlighted_html = text

    top_words_html = ""
    for w in top_words:
        is_pos  = w["score"] > 0
        color   = verdict_color if is_pos else ("#52C9A0" if is_deceptive else "#F0527A")
        bg      = f"rgba(240,82,122,0.10)" if (is_pos and is_deceptive) else \
                  f"rgba(82,201,160,0.10)" if (is_pos and not is_deceptive) else \
                  f"rgba(82,201,160,0.10)" if is_deceptive else f"rgba(240,82,122,0.10)"
        arrow   = "&#8593;" if is_pos else "&#8595;"
        top_words_html += f"""
        <div class="word-pill">
          <span class="word-text">{w["word"]}</span>
          <span class="word-score" style="color:{color};background:{bg}">
            {arrow} {abs(w["score"]):.2f}
          </span>
        </div>"""

    conf_offset = 251.2 * (1 - float(confidence)/100) if confidence else 251.2


    # Pre-build conditional HTML blocks — avoids backslash-in-f-string errors
    if highlighted_html:
        rc = (
            '<div class="review-card">'
            '<p class="sec-title">Review Text &#8212; Word Influence Highlight</p>'
            '<div class="review-text">' + highlighted_html + '</div>'
            '<div class="legend-row">'
            '<span><span class="legend-dot" style="background:' + verdict_color + ';opacity:.7"></span>Toward ' + label + '</span>'
            '<span><span class="legend-dot" style="background:' + ('#52C9A0' if is_deceptive else '#F0527A') + ';opacity:.6"></span>Against ' + label + '</span>'
            '<span style="margin-left:auto;font-size:11px">Intensity = influence strength</span>'
            '</div></div>'
        )
    else:
        rc = ''

    if top_words_html:
        wc = (
            '<div class="words-card">'
            '<p class="sec-title">Top Influential Words</p>'
            '<div class="words-grid">' + top_words_html + '</div>'
            '</div>'
        )
    else:
        wc = ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RyView &#8212; Analysis Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{
      --bg:      #110D1A;
      --s1:      #1C1628;
      --s2:      #241E33;
      --border:  rgba(255,255,255,0.08);
      --text:    #EAE6F0;
      --muted:   #7A7290;
      --accent:  #9B6DFF;
      --verdict: {verdict_color};
      --vbg:     {verdict_bg};
      --font-h:  'Syne', sans-serif;
      --font-b:  'DM Sans', sans-serif;
    }}
    html{{scroll-behavior:smooth}}
    body{{
      background:var(--bg);color:var(--text);
      font-family:var(--font-b);font-size:15px;
      line-height:1.6;min-height:100vh;
    }}
    .blob{{
      position:fixed;border-radius:50%;
      filter:blur(130px);pointer-events:none;z-index:0;
    }}
    .blob-1{{
      width:550px;height:550px;
      background:{"rgba(240,82,122,0.07)" if is_deceptive else "rgba(82,201,160,0.06)"};
      top:-180px;right:-180px;
      animation:drift 14s ease-in-out infinite alternate;
    }}
    .blob-2{{
      width:450px;height:450px;
      background:rgba(155,109,255,0.06);
      bottom:-120px;left:-120px;
      animation:drift 14s ease-in-out infinite alternate;
      animation-delay:-7s;
    }}
    @keyframes drift{{
      from{{transform:translate(0,0) scale(1)}}
      to{{transform:translate(28px,18px) scale(1.05)}}
    }}
    .page{{
      position:relative;z-index:1;
      max-width:800px;margin:0 auto;
      padding:44px 24px 80px;
    }}
    .header{{
      display:flex;align-items:center;
      justify-content:space-between;
      margin-bottom:40px;
      animation:fadeUp .4s ease both;
    }}
    .brand{{display:flex;align-items:center;gap:10px;}}
    .brand-icon{{
      width:36px;height:36px;border-radius:10px;
      background:linear-gradient(135deg,var(--accent),#7C3AED);
      display:flex;align-items:center;justify-content:center;
      font-size:18px;font-weight:800;color:white;
      font-family:var(--font-h);
    }}
    .brand-name{{font-family:var(--font-h);font-size:18px;font-weight:800;}}
    .status-pill{{
      display:flex;align-items:center;gap:6px;
      padding:6px 14px;border-radius:999px;
      border:1px solid var(--border);
      background:var(--s1);
      font-size:12px;color:var(--muted);
    }}
    .pulse{{
      width:7px;height:7px;border-radius:50%;
      background:#52C9A0;box-shadow:0 0 6px #52C9A0;
      animation:pulse 2s infinite;
    }}
    @keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.4;transform:scale(1.4)}}}}
    .verdict-card{{
      background:var(--s1);
      border:1px solid var(--border);
      border-radius:24px;
      padding:32px 36px;
      margin-bottom:16px;
      position:relative;overflow:hidden;
      animation:fadeUp .4s .08s ease both;
    }}
    .verdict-card::before{{
      content:'';
      position:absolute;top:0;left:0;right:0;height:2px;
      background:linear-gradient(90deg,transparent,var(--verdict),transparent);
    }}
    .verdict-top{{
      display:flex;align-items:center;
      justify-content:space-between;
      margin-bottom:22px;
    }}
    .eyebrow{{
      font-size:11px;font-weight:500;
      letter-spacing:.1em;text-transform:uppercase;
      color:var(--muted);
    }}
    .verdict-badge{{
      display:flex;align-items:center;gap:7px;
      background:var(--vbg);
      border:1px solid var(--verdict);
      border-radius:999px;padding:5px 15px;
      font-family:var(--font-h);font-size:13px;font-weight:700;
      color:var(--verdict);
    }}
    .verdict-main{{
      display:flex;align-items:flex-end;
      justify-content:space-between;gap:16px;
    }}
    .verdict-label{{
      font-family:var(--font-h);
      font-size:52px;font-weight:800;
      line-height:1;letter-spacing:-2px;
      color:var(--verdict);
    }}
    .verdict-desc{{
      font-size:13px;color:var(--muted);
      margin-top:8px;max-width:340px;line-height:1.6;
    }}
    .conf-ring{{
      flex-shrink:0;position:relative;
      width:96px;height:96px;
    }}
    .conf-ring svg{{transform:rotate(-90deg);}}
    .ring-track{{fill:none;stroke:var(--s2);stroke-width:8;}}
    .ring-fill{{
      fill:none;stroke:var(--verdict);
      stroke-width:8;stroke-linecap:round;
      stroke-dasharray:251.2;
      stroke-dashoffset:{conf_offset};
    }}
    .conf-center{{
      position:absolute;inset:0;
      display:flex;flex-direction:column;
      align-items:center;justify-content:center;
    }}
    .conf-val{{font-family:var(--font-h);font-size:19px;font-weight:800;line-height:1;}}
    .conf-lbl{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;}}
    .sec-title{{
      font-size:11px;font-weight:600;
      letter-spacing:.1em;text-transform:uppercase;
      color:var(--muted);margin-bottom:12px;
    }}
    .review-card{{
      background:var(--s1);border:1px solid var(--border);
      border-radius:20px;padding:24px 28px;
      margin-bottom:16px;
      animation:fadeUp .4s .16s ease both;
    }}
    .review-text{{
      font-size:14.5px;line-height:1.85;
      color:var(--text);word-break:break-word;
    }}
    .legend-row{{
      display:flex;align-items:center;gap:16px;
      margin-top:14px;padding-top:12px;
      border-top:1px solid var(--border);
      font-size:12px;color:var(--muted);
    }}
    .legend-dot{{
      width:10px;height:10px;border-radius:3px;
      display:inline-block;margin-right:5px;
    }}
    .words-card{{
      background:var(--s1);border:1px solid var(--border);
      border-radius:20px;padding:22px 28px;
      margin-bottom:16px;
      animation:fadeUp .4s .22s ease both;
    }}
    .words-grid{{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px;}}
    .word-pill{{
      display:flex;align-items:center;gap:7px;
      background:var(--s2);border:1px solid var(--border);
      border-radius:999px;padding:5px 13px;
    }}
    .word-text{{font-size:13px;font-weight:500;color:var(--text);}}
    .word-score{{font-size:11px;font-weight:700;padding:2px 7px;border-radius:999px;}}
    .explain-card{{
      background:var(--s1);border:1px solid var(--border);
      border-radius:20px;padding:22px 28px;
      margin-bottom:16px;
      animation:fadeUp .4s .28s ease both;
      font-size:13.5px;color:var(--muted);line-height:1.7;
    }}
    .explain-card strong{{color:var(--text);}}
    .footer{{
      text-align:center;font-size:12px;color:var(--muted);
      margin-top:44px;animation:fadeUp .4s .34s ease both;
    }}
    @keyframes fadeUp{{
      from{{opacity:0;transform:translateY(14px)}}
      to{{opacity:1;transform:translateY(0)}}
    }}
    @media(max-width:560px){{
      .verdict-label{{font-size:36px;}}
      .verdict-card{{padding:22px 18px;}}
    }}
  </style>
</head>
<body>
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>
  <div class="page">

    <header class="header">
      <div class="brand">
        <div class="brand-icon">&#119825;</div>
        <div class="brand-name">RyView</div>
      </div>
      <div class="status-pill">
        <div class="pulse"></div>
        Analysis Complete
      </div>
    </header>

    <div class="verdict-card">
      <div class="verdict-top">
        <div class="eyebrow">Analysis Verdict</div>
        <div class="verdict-badge">{verdict_icon}&nbsp;{label if label else "&#8212;"}</div>
      </div>
      <div class="verdict-main">
        <div>
          <div class="verdict-label">{label if label else "&#8212;"}</div>
          <div class="verdict-desc">
            {"This review contains language patterns the model associates with deceptive content. The highlighted words below show what influenced this decision." if is_deceptive else "This review contains language patterns the model associates with genuine content. The highlighted words below show what influenced this decision."}
          </div>
        </div>
        <div class="conf-ring">
          <svg width="96" height="96" viewBox="0 0 100 100">
            <circle class="ring-track" cx="50" cy="50" r="40"/>
            <circle class="ring-fill"  cx="50" cy="50" r="40"/>
          </svg>
          <div class="conf-center">
            <div class="conf-val">{conf_display}</div>
            <div class="conf-lbl">Confidence</div>
          </div>
        </div>
      </div>
    </div>

    {rc}

    {wc}

    <div class="explain-card">
      <strong>How this verdict was reached</strong><br/>
      The review was processed through RoBERTa&#8217;s 12 transformer layers, building a contextual understanding of each word relative to the full text. Classification was made by a linear layer on the [CLS] token embedding. The highlights above use <strong>Gradient &times; Input saliency</strong> &#8212; measuring how much each word pushed the model toward or away from the <strong>{label}</strong> verdict. Darker highlights indicate stronger influence.
    </div>

    <div class="footer">
      Powered by <strong>RyView</strong> &nbsp;&middot;&nbsp;
      Enhanced RoBERTa for Deceptive Hotel Review Detection
    </div>

  </div>
</body>
</html>"""
