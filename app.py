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

    return f"""
    <!DOCTYPE html>
    <html>

    <head>
        <title>AI Review Analysis</title>

        <style>

            body {{
                font-family: "Segoe UI", Arial, sans-serif;
                background: linear-gradient(to right, #f5f7fa, #e4ecf5);
                margin: 0;
                padding: 40px;
            }}

            .container {{
                max-width: 700px;
                margin: auto;
                background: white;
                padding: 35px;
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.12);
            }}

            h1 {{
                text-align: center;
                margin-bottom: 30px;
            }}

            .prediction-card {{
                padding: 20px;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin-bottom: 30px;
                background: {"linear-gradient(135deg, #e74c3c, #c0392b)" if label == "Deceptive" else "linear-gradient(135deg, #2ecc71, #27ae60)"};
            }}

            .prediction-label {{
                font-size: 28px;
                font-weight: bold;
            }}

            .confidence {{
                font-size: 18px;
                margin-top: 10px;
            }}

            .metrics {{
                margin-top: 20px;
            }}

            .metric {{
                background: #f7f9fc;
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 12px;
                display: flex;
                justify-content: space-between;
                font-size: 16px;
            }}

        </style>

    </head>

    <body>

        <div class="container">

            <h1>🧠 AI Review Analysis</h1>

            <div class="prediction-card">

                <div class="prediction-label">
                    {label}
                </div>

                <div class="confidence">
                    Confidence: {confidence}%
                </div>

            </div>

            <h2>📊 Model Performance</h2>

            <div class="metrics">

                <div class="metric">
                    <span>Accuracy</span>
                    <span>89.58%</span>
                </div>

                <div class="metric">
                    <span>Precision</span>
                    <span>84.17%</span>
                </div>

                <div class="metric">
                    <span>Recall</span>
                    <span>97.50%</span>
                </div>

                <div class="metric">
                    <span>F1-Score</span>
                    <span>90.35%</span>
                </div>

            </div>

        </div>

    </body>

    </html>
    """