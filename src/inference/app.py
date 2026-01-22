from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch

app = FastAPI(title="DocuCode API")

# Load model once at startup
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "models/docucode_adapter", # Your saved adapter path
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

class CodeRequest(BaseModel):
    code: str
    style: str = "Google"

@app.post("/generate_comment")
async def generate(request: CodeRequest):
    prompt = f"### Instruction:\nGenerate a {request.style} style docstring.\n\n### Code:\n{request.code}\n\n### Response:\n"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return {"docstring": response.split("### Response:\n")[-1]}