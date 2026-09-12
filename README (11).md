# Dialogue Summarizer

A web application that summarizes multi-turn dialogues into a single clean summary line, using a fine-tuned T5 model served through a FastAPI backend.

<img width="1900" height="917" alt="Screenshot 2026-09-12 163941" src="https://github.com/user-attachments/assets/757240bc-927b-4680-85b7-aa3e368f4c4f" />

## Features

- **Dialogue summarization** — condenses back-and-forth conversations into one concise summary.
- **Text cleaning pipeline** — strips line breaks, extra whitespace, and HTML tags, and normalizes text before feeding it to the model.
- **Beam search generation** — generates the summary using 4 beams with early stopping for higher-quality output.
- **Automatic device selection** — runs on GPU (CUDA) if available, otherwise falls back to CPU.
- **REST API**
  - `GET /` — serves the web UI.
  - `POST /summarize/` — accepts a dialogue string and returns the generated summary as JSON.
- **Request validation** — incoming requests are validated using a Pydantic schema (`DialogueInput`).
- **Web interface**
  - Text area for pasting a conversation, with live token count.
  - Output panel displaying the generated summary with a typewriter effect.
  - Compression gauge showing the ratio between input and output length.
  - Preloaded example dialogues for quick testing.
  - Automatic fallback to a local demo summary if the backend is unreachable.

## Frameworks & Libraries Used

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) — Python web framework for building the API.
- [Uvicorn](https://www.uvicorn.org/) — ASGI server used to run the FastAPI app.
- [Pydantic](https://docs.pydantic.dev/) — request body validation.
- [Jinja2Templates](https://jinja.palletsprojects.com/) (via FastAPI) — HTML templating for serving the UI.
- [PyTorch](https://pytorch.org/) — tensor computation and model execution.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — `T5ForConditionalGeneration` and `T5Tokenizer` for loading and running the summarization model.
- `re` (Python standard library) — text cleaning via regular expressions.

**Frontend**
- HTML5, CSS3, and vanilla JavaScript (no frontend framework).
- Google Fonts — Inter and IBM Plex Mono.

**Model**
- A fine-tuned **T5** (Text-to-Text Transfer Transformer) model, loaded from a local directory (`./saved_summary_model`).

## Project Structure

```
TEXTSUMMARIZER/
├── app.py                  # FastAPI application and summarization logic
├── index.html               # Frontend web UI
└── saved_summary_model/     # Fine-tuned T5 model and tokenizer files
```

## Running the App

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

## API Usage

**Endpoint:** `POST /summarize/`

**Request body:**
```json
{
  "dialogue": "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I'll bring you tomorrow :-)"
}
```

**Response:**
```json
{
  "summary": "Amanda made cookies and will bring some to Jerry tomorrow."
}
```
