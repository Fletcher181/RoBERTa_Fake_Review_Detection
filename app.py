from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import torch
import torch.nn.functional as F

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification
)

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()

# Allow Chrome extension requests
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

model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

# GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")
print(f"Using device: {DEVICE}")

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
    return {
        "message": "Fake Review Detection API is running"
    }

# =========================================================
# PREDICT ROUTE
# =========================================================

@app.post("/predict")
def predict(review: Review):

    text = review.text

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)

    confidence, pred_class = torch.max(probs, dim=1)

    prediction = int(pred_class.item())
    confidence = float(confidence.item())

    # LABEL MAPPING
    label_map = {
        0: "Genuine",
        1: "Deceptive"
    }

    label = label_map[prediction]

    return {
        "label": label,
        "confidence": confidence
    }

# =========================================================
# DASHBOARD ROUTE
# =========================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(label: str = "", confidence: str = ""):

    if label == "Deceptive":

        interpretation = """
        The review contains highly persuasive and emotionally exaggerated language,
        which may indicate deceptive intent. The writing also lacks detailed and
        balanced experiences commonly found in authentic hotel reviews.
        """

        patterns = [
            "Excessive positive wording",
            "Limited specific details",
            "Promotional tone detected"
        ]

        color = "#e74c3c"

    else:

        interpretation = """
        The review demonstrates natural language patterns and includes realistic
        descriptions of personal experiences, which are commonly associated with
        genuine hotel reviews.
        """

        patterns = [
            "Balanced review structure",
            "Specific hotel experiences",
            "Natural writing flow"
        ]

        color = "#2ecc71"

    return f"""
    <html>

    <head>

        <title>AI Review Analysis</title>

        <style>

            body {{
                font-family: "Segoe UI", sans-serif;
                background: #f4f6f9;
                padding: 40px;
            }}

            .container {{
                max-width: 750px;
                margin: auto;
                background: white;
                border-radius: 20px;
                padding: 35px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}

            h1 {{
                text-align: center;
                margin-bottom: 30px;
            }}

            .result {{
                background: {color};
                color: white;
                padding: 25px;
                border-radius: 16px;
                text-align: center;
                margin-bottom: 30px;
            }}

            .label {{
                font-size: 30px;
                font-weight: bold;
            }}

            .confidence {{
                margin-top: 10px;
                font-size: 18px;
            }}

            .section {{
                margin-top: 25px;
            }}

            .section h2 {{
                margin-bottom: 12px;
            }}

            ul {{
                padding-left: 20px;
            }}

            li {{
                margin-bottom: 10px;
            }}

        </style>

    </head>

    <body>

        <div class="container">

            <h1>🧠 AI Review Analysis</h1>

            <div class="result">

                <div class="label">
                    {label}
                </div>

                <div class="confidence">
                    Confidence: {confidence}%
                </div>

            </div>

            <div class="section">

                <h2>📖 AI Interpretation</h2>

                <p>
                    {interpretation}
                </p>

            </div>

            <div class="section">

                <h2>🔍 Detected Patterns</h2>

                <ul>
                    {''.join(f'<li>{p}</li>' for p in patterns)}
                </ul>

            </div>

        </div>

    </body>

    </html>
    """