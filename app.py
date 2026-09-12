#fastapi - python based webframework for building APIs
import re

from fastapi import FastAPI,Request
from pydantic import BaseModel #python module used to validate the request body

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

#initialize the FastAPI app
app = FastAPI(title="Text Summarization APP", description="This is a text summarization APP built using FastAPI and T5 model.", version="1.0.0")

#Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

#DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#templating
templates = Jinja2Templates(directory=".")

#input schema for the dialogue =>string
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text=re.sub(r"\r\n"," ",text) #lines
    text=re.sub(r"\s+"," ",text) #Spaces
    text=re.sub(r"<.*?>"," ",text) #html tags
    text=text.strip().lower()
    return text

def summarize_dialogue(dialogue:str)->str:
    dialogue=clean_data(dialogue) # clean

    #tokenize
    inputs=tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt" # pytorch tensors
    ).to(device)
    
    #generate the summary => tokens id
    model.to(device)
    targets=model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4, # it will generate 4 diff outputs and it will give the best among that
        early_stopping=True
    )

    #tokens id convert to text => Decoding
    summary=tokenizer.decode(targets[0],skip_special_tokens=True) #EOS,SEP,SPACES
    return summary

#API endpoint for summarization

#GET API
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}