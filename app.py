from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI()

MODEL_PATH = "./roberta_fake_review_model"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
model.eval()

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict(review: Review):
    inputs = tokenizer(
        review.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probs = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    label_map = {0: "Genuine", 1: "Deceptive"}

    return {
        "label": label_map[pred_class.item()],
        "confidence": round(confidence.item(), 4)
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)