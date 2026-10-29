import os
import re

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_summary_model")
MODEL_FILE = os.path.join(MODEL_DIR, "model.safetensors")

app = FastAPI(
    title="Text Summarization APP",
    description="Summarizes dialogues with a fine-tuned T5 model.",
    version="1.0.0",
)

# Fail at startup with a clear message if the model is missing or is only a Git LFS pointer.
if not os.path.isfile(MODEL_FILE):
    raise RuntimeError(
        f"model.safetensors not found in {MODEL_DIR}. "
        "Check that the model was committed and pulled with Git LFS."
    )
if os.path.getsize(MODEL_FILE) < 1_000_000:
    raise RuntimeError(
        "model.safetensors is a small Git LFS pointer, not the real model. "
        "Run 'git lfs pull' or check the LFS upload."
    )

tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, low_cpu_mem_usage=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

templates = Jinja2Templates(directory=BASE_DIR)


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
    text = re.sub(r"\s+", " ", text)      # collapse line breaks and extra spaces
    return text.strip()


def summarize_dialogue(dialogue: str) -> str:
    cleaned = clean_data(dialogue)

    inputs = tokenizer(
        cleaned,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}


# Plain def (not async) so the blocking model call runs in a worker thread.
@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):
    if not dialogue_input.dialogue.strip():
        raise HTTPException(status_code=400, detail="Dialogue is empty.")
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}