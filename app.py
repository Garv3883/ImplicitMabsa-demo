from __future__ import annotations

import io
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import Any, Dict

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from dataset import (
    SENTIMENT_LABELS,
    DataConfig,
    load_image_processor,
    load_tokenizer,
)
from model import (
    STREAM_DISPLAY_NAMES,
    VALID_MODES,
    ImplicitMABSAModel,
    ModelConfig,
)
from utils import get_device, load_checkpoint

CHECKPOINT_PATH = Path("saved_models/implicit_mabsa.pt")

app = FastAPI(title="ImplicitMABSA Live Demo", version="1.0.0")

# Global state loaded once at startup
STATE: Dict[str, Any] = {}


@app.on_event("startup")
def load_models() -> None:
    if not CHECKPOINT_PATH.is_file():
        hf_hub_download(
            repo_id="imdb",
            filename="implicit_mabsa.pt",
            local_dir="saved_models/"
        )   

    device = get_device(None)
    ckpt = load_checkpoint(str(CHECKPOINT_PATH))
    model_cfg = ModelConfig(**ckpt["model_config"])
    data_cfg = DataConfig(**ckpt["data_config"])

    tokenizer = load_tokenizer(model_cfg.text_model_name)
    image_processor = load_image_processor(model_cfg.vision_model_name)

    models: Dict[str, ImplicitMABSAModel] = {}
    for mode in VALID_MODES:
        if mode in ckpt["streams"]:
            m = ImplicitMABSAModel(model_cfg, modes=(mode,), pretrained=False)
            m.load_stream_state_dict(mode, ckpt["streams"][mode])
            m.to(device)
            m.eval()
            models[mode] = m

    STATE["device"] = device
    STATE["tokenizer"] = tokenizer
    STATE["image_processor"] = image_processor
    STATE["max_length"] = data_cfg.max_length
    STATE["models"] = models
    STATE["aspect_vocab"] = ckpt.get("aspect_vocab", [])


@torch.inference_mode()
def run_inference(text: str, aspect: str, image: Image.Image) -> Dict[str, Any]:
    device: torch.device = STATE["device"]
    tokenizer = STATE["tokenizer"]
    image_processor = STATE["image_processor"]
    max_length: int = STATE["max_length"]
    models: Dict[str, ImplicitMABSAModel] = STATE["models"]

    enc = tokenizer(
        text.strip(),
        aspect.strip().lower(),
        truncation="only_first",
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc["token_type_ids"].to(device) if "token_type_ids" in enc else None

    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    predictions: Dict[str, Any] = {}
    for mode, model in models.items():
        kwargs: Dict[str, Any] = {}
        if mode in ("text_only", "multimodal"):
            kwargs["input_ids"] = input_ids
            kwargs["attention_mask"] = attention_mask
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
        if mode in ("image_only", "multimodal"):
            kwargs["pixel_values"] = pixel_values

        out = model(mode, **kwargs)
        sent_probs = F.softmax(out["sentiment_logits"].float(), dim=-1).squeeze(0).cpu().tolist()
        pred_idx = int(torch.tensor(sent_probs).argmax().item())

        aspect_info = None
        if out["aspect_logits"] is not None:
            asp_probs = F.softmax(out["aspect_logits"].float(), dim=-1).squeeze(0).cpu().tolist()
            aspect_info = {
                "is_relevant": bool(asp_probs[1] >= 0.5),
                "relevance_score": round(asp_probs[1] * 100, 2),
            }

        predictions[mode] = {
            "stream_name": STREAM_DISPLAY_NAMES[mode],
            "predicted_sentiment": SENTIMENT_LABELS[pred_idx],
            "confidence": round(sent_probs[pred_idx] * 100, 2),
            "probabilities": {
                label: round(prob * 100, 2)
                for label, prob in zip(SENTIMENT_LABELS, sent_probs)
            },
            "aspect_relevance": aspect_info,
        }

    return {
        "input": {"text": text, "aspect": aspect},
        "device_used": str(device),
        "results": predictions,
    }


@app.post("/predict")
async def predict_endpoint(
    text: str = Form(..., description="Tweet or post text"),
    aspect: str = Form(..., description="Target or candidate aspect"),
    image: UploadFile = File(..., description="Accompanying image file"),
) -> Dict[str, Any]:
    try:
        raw_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    return run_inference(text=text, aspect=aspect, image=pil_img)


@app.get("/", response_class=HTMLResponse)
def demo_ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ImplicitMABSA — Live Multimodal Sentiment Demo</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 820px; margin: 40px auto; padding: 0 20px; background: #0f172a; color: #f8fafc; }
    h1 { font-size: 1.6rem; margin-bottom: 6px; }
    p.sub { color: #94a3b8; margin-top: 0; margin-bottom: 24px; }
    .card { background: #1e293b; padding: 22px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; }
    label { display: block; font-weight: 600; margin-bottom: 6px; color: #cbd5e1; }
    input[type="text"], textarea, input[type="file"] {
      width: 100%; box-sizing: border-box; padding: 10px 12px; border-radius: 8px;
      border: 1px solid #475569; background: #0f172a; color: #f8fafc; margin-bottom: 16px; font-size: 0.95rem;
    }
    button {
      background: #3b82f6; color: white; border: none; padding: 12px 20px; border-radius: 8px;
      font-weight: 600; font-size: 1rem; cursor: pointer; width: 100%;
    }
    button:hover { background: #2563eb; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-top: 16px; }
    .stream-box { background: #0f172a; padding: 16px; border-radius: 10px; border: 1px solid #334155; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; }
    .positive { background: #065f46; color: #6ee7b7; }
    .neutral { background: #78350f; color: #fcd34d; }
    .negative { background: #7f1d1d; color: #fca5a5; }
    .meta { font-size: 0.85rem; color: #94a3b8; margin-top: 8px; line-height: 1.5; }
  </style>
</head>
<body>
  <h1>ImplicitMABSA Live Inference</h1>
  <p class="sub">Compare Text-Only (DeBERTa-v3), Image-Only (ViT), and Cross-Modal Fusion live.</p>

  <div class="card">
    <form id="predictForm">
      <label for="text">Tweet / Post Text</label>
      <textarea id="text" name="text" rows="3" placeholder="e.g., Well, that happened after two hours of use..." required></textarea>

      <label for="aspect">Target / Candidate Aspect</label>
      <input type="text" id="aspect" name="aspect" placeholder="e.g., battery, screen, food, camera" required />

      <label for="image">Accompanying Image</label>
      <input type="file" id="image" name="image" accept="image/*" required />

      <button type="submit" id="submitBtn">Analyze Multimodal Sentiment</button>
    </form>
  </div>

  <div id="output" style="display:none;" class="card">
    <h3 style="margin-top:0;">Stream-by-Stream Comparison</h3>
    <div class="grid" id="resultsGrid"></div>
  </div>

  <script>
    const form = document.getElementById("predictForm");
    const btn = document.getElementById("submitBtn");
    const outBox = document.getElementById("output");
    const grid = document.getElementById("resultsGrid");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      btn.disabled = true;
      btn.textContent = "Running DeBERTa + ViT Inference...";
      grid.innerHTML = "";

      try {
        const formData = new FormData(form);
        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Prediction failed");

        for (const key of ["text_only", "image_only", "multimodal"]) {
          const item = data.results[key];
          if (!item) continue;
          const s = item.predicted_sentiment;
          const aspHtml = item.aspect_relevance
            ? `<div>Aspect Match: <b>${item.aspect_relevance.is_relevant ? "Yes" : "No"}</b> (${item.aspect_relevance.relevance_score}%)</div>`
            : `<div>Aspect Match: <i>n/a (vision only)</i></div>`;

          grid.innerHTML += `
            <div class="stream-box">
              <div style="font-weight:700; margin-bottom:10px;">${item.stream_name}</div>
              <span class="badge ${s}">${s} (${item.confidence}%)</span>
              <div class="meta">
                ${aspHtml}
                <div style="margin-top:6px;">
                  Pos: ${item.probabilities.positive}% | Neu: ${item.probabilities.neutral}% | Neg: ${item.probabilities.negative}%
                </div>
              </div>
            </div>`;
        }
        outBox.style.display = "block";
      } catch (err) {
        alert(err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = "Analyze Multimodal Sentiment";
      }
    });
  </script>
</body>
</html>"""