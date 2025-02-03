from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TextIteratorStreamer,
    AutoConfig
)
from peft import PeftModel, PeftConfig
import torch
from typing import Dict, Optional
import gc
import asyncio
from threading import Thread
import json
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add these near the top with your other imports
templates = Jinja2Templates(directory="templates")

class ModelConfig(BaseModel):
    base_model_name: str = "facebook/opt-125m"  # Small model for testing
    lora_adapter_path: Optional[str] = None
    max_memory: Optional[Dict[str, str]] = None  # Memory management config

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        # Use MPS (Metal Performance Shaders) if available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def _clear_memory(self):
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            if self.device == "mps":
                torch.mps.empty_cache()
            gc.collect()

    def load_model(self, model_config: ModelConfig):
        self._clear_memory()

        try:
            logger.info(f"Starting to load model: {model_config.base_model_name}")
            
            # Modified device mapping
            if self.device == "mps":
                logger.info("Loading model to CPU first, then will move to MPS")
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.base_model_name,
                    torch_dtype=torch.float16,
                    device_map={"": "cpu"},
                    trust_remote_code=True
                )
                logger.info("Moving model to MPS device")
                model = model.to(self.device)
            else:
                logger.info("Loading model directly to CPU")
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.base_model_name,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    trust_remote_code=True
                )
            
            logger.info("Loading tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.base_model_name,
                use_fast=True,
                trust_remote_code=True
            )
            
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_name = model_config.base_model_name
            
            logger.info(f"Successfully loaded model: {model_config.base_model_name}")
            return {
                "status": "success", 
                "message": f"Loaded model: {model_config.base_model_name}",
                "device": self.device
            }
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error loading model: {str(e)}\nType: {type(e).__name__}"
            )

model_manager = ModelManager()

@app.post("/load_model")
async def load_model(model_config: ModelConfig):
    try:
        return model_manager.load_model(model_config)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in load_model endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current_model")
async def get_current_model():
    if model_manager.current_model_name:
        return {
            "current_model": model_manager.current_model_name,
            "device": model_manager.device
        }
    return {"current_model": None}

# Add this new class for the request body
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    if not model_manager.current_model:
        raise HTTPException(status_code=400, detail="No model currently loaded")
    
    try:
        inputs = model_manager.current_tokenizer(
            request.prompt, 
            return_tensors="pt"
        ).to(model_manager.device)
        
        outputs = model_manager.current_model.generate(
            **inputs,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=1,
            pad_token_id=model_manager.current_tokenizer.eos_token_id
        )
        
        response = model_manager.current_tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        return {"generated_text": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    if not model_manager.current_model:
        raise HTTPException(status_code=400, detail="No model currently loaded")
    
    async def generate_tokens():
        try:
            inputs = model_manager.current_tokenizer(
                request.prompt, 
                return_tensors="pt"
            ).to(model_manager.device)
            
            # Enable streaming generation
            streamer = TextIteratorStreamer(
                model_manager.current_tokenizer,
                skip_special_tokens=True
            )
            
            generation_kwargs = dict(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                streamer=streamer,
                pad_token_id=model_manager.current_tokenizer.eos_token_id
            )

            # Run generation in a separate thread
            thread = Thread(target=model_manager.current_model.generate, 
                          kwargs=generation_kwargs)
            thread.start()

            # Stream the output tokens
            for text in streamer:
                yield f"data: {json.dumps({'token': text})}\n\n"
                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming the client
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream"
    )

# Add this route to serve the HTML page
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
