"""Lightweight inference API for ImplicitMABSA.

    from inference import predict_sentiment
    result = predict_sentiment("It dies after two hours", "post.jpg", ["battery", "screen", "food"])

For every candidate aspect the text and multimodal streams output an aspect-relevance
probability and a polarity distribution; the candidate with the highest relevance is
reported as the target implicit aspect. The image-only stream never sees the candidate
text, so it returns a single polarity for the image (``target_aspect`` is ``None``).

Candidate aspects outside the training vocabulary are accepted (they are just text) but
the models were only trained on the categories in ``aspect_vocab``.

CLI:
    python inference.py --text "It dies after two hours" --image post.jpg --aspects battery screen
    python inference.py --demo
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

from dataset import load_image_processor, load_tokenizer, render_synthetic_image
from model import VALID_MODES, ImplicitMABSAModel, ModelConfig, select_inputs
from utils import DEFAULT_CHECKPOINT_PATH, get_device, load_checkpoint, setup_logging

logger = logging.getLogger(__name__)


class ImplicitMABSAPredictor:
    """Loads the fine-tuned checkpoint once and serves predictions for all trained streams."""

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, device: Optional[str] = None) -> None:
        self.device = get_device(device)
        ckpt = load_checkpoint(checkpoint_path)
        self.model_config = ModelConfig(**ckpt["model_config"])
        self.max_length: int = int(ckpt["data_config"]["max_length"])
        self.sentiment_labels: List[str] = list(ckpt["sentiment_labels"])
        self.aspect_vocab: List[str] = list(ckpt["aspect_vocab"])

        self.tokenizer = load_tokenizer(self.model_config.text_model_name)
        self.image_processor = load_image_processor(self.model_config.vision_model_name)

        # One independent model per stream (each owns its fine-tuned encoder copies).
        self.models: Dict[str, ImplicitMABSAModel] = {}
        for mode in VALID_MODES:
            if mode not in ckpt["streams"]:
                logger.warning("Stream %s is not in the checkpoint; it will be omitted.", mode)
                continue
            model = ImplicitMABSAModel(self.model_config, modes=(mode,), pretrained=False)
            model.load_stream_state_dict(mode, ckpt["streams"][mode])
            self.models[mode] = model.to(self.device).eval()
        logger.info("Predictor ready on %s with streams: %s", self.device, list(self.models))

    # ------------------------------------------------------------------ #
    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float(), dim=-1).cpu()

    def _polarity_distribution(self, probs: torch.Tensor) -> Dict[str, float]:
        return {label: round(float(p), 4) for label, p in zip(self.sentiment_labels, probs)}

    def _summarise_text_stream(
        self, outputs: Dict[str, Optional[torch.Tensor]], aspects: Sequence[str]
    ) -> Dict[str, Any]:
        sent_probs = self._probs(outputs["sentiment_logits"])  # [K, 3]
        relevance = self._probs(outputs["aspect_logits"])[:, 1]  # [K]
        per_aspect = []
        for i, aspect in enumerate(aspects):
            top = int(sent_probs[i].argmax())
            per_aspect.append(
                {
                    "aspect": aspect,
                    "relevance_confidence": round(float(relevance[i]), 4),
                    "sentiment": self.sentiment_labels[top],
                    "sentiment_confidence": round(float(sent_probs[i][top]), 4),
                }
            )
        best = int(relevance.argmax())
        return {
            "target_aspect": aspects[best],
            "aspect_confidence": per_aspect[best]["relevance_confidence"],
            "sentiment": per_aspect[best]["sentiment"],
            "sentiment_confidence": per_aspect[best]["sentiment_confidence"],
            "sentiment_probabilities": self._polarity_distribution(sent_probs[best]),
            "per_aspect": per_aspect,
        }

    def _summarise_image_stream(self, outputs: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Any]:
        probs = self._probs(outputs["sentiment_logits"])[0]  # identical for every candidate row
        top = int(probs.argmax())
        return {
            "target_aspect": None,
            "aspect_confidence": None,
            "sentiment": self.sentiment_labels[top],
            "sentiment_confidence": round(float(probs[top]), 4),
            "sentiment_probabilities": self._polarity_distribution(probs),
            "per_aspect": [],
        }

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict_sentiment(
        self, text: str, image_path: str, candidate_aspects: List[str]
    ) -> Dict[str, Any]:
        """Predict target implicit aspect and polarity with all available streams.

        Args:
            text: the review / post text.
            image_path: path to the accompanying image.
            candidate_aspects: candidate aspect categories to rank.

        Returns:
            JSON-serialisable dict::

                {"text": ..., "image_path": ..., "candidate_aspects": [...],
                 "streams": {"text_only": {...}, "image_only": {...}, "multimodal": {...}}}
        """
        aspects = [a.strip() for a in candidate_aspects if isinstance(a, str) and a.strip()]
        if not aspects:
            raise ValueError("candidate_aspects must contain at least one non-empty string.")
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string.")

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.expand(len(aspects), -1, -1, -1)  # one image, K candidates

        enc = self.tokenizer(
            [text] * len(aspects),
            aspects,  # pair encoding -> [CLS] text [SEP] aspect [SEP]
            truncation="only_first",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        batch: Dict[str, torch.Tensor] = {k: v for k, v in enc.items()}
        batch["pixel_values"] = pixel_values

        streams: Dict[str, Optional[Dict[str, Any]]] = {mode: None for mode in VALID_MODES}
        for mode, model in self.models.items():
            outputs = model(mode, **select_inputs(batch, mode, self.device))
            if mode == "image_only":
                streams[mode] = self._summarise_image_stream(outputs)
            else:
                streams[mode] = self._summarise_text_stream(outputs, aspects)

        return {
            "text": text,
            "image_path": str(image_path),
            "candidate_aspects": aspects,
            "streams": streams,
        }


_PREDICTORS: Dict[str, ImplicitMABSAPredictor] = {}


def predict_sentiment(
    text: str,
    image_path: str,
    candidate_aspects: List[str],
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
) -> Dict[str, Any]:
    """Module-level API function (the predictor is created lazily and cached per checkpoint)."""
    key = str(Path(checkpoint_path).resolve())
    if key not in _PREDICTORS:
        _PREDICTORS[key] = ImplicitMABSAPredictor(checkpoint_path)
    return _PREDICTORS[key].predict_sentiment(text, image_path, candidate_aspects)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="ImplicitMABSA inference.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--text", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--aspects", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--demo", action="store_true", help="run on a generated synthetic battery example")
    args = parser.parse_args()

    predictor = ImplicitMABSAPredictor(args.checkpoint, args.device)
    text, image_path, aspects = args.text, args.image, args.aspects

    if args.demo:
        demo_dir = Path(args.checkpoint).parent
        demo_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(demo_dir / "demo_battery_negative.png")
        render_synthetic_image("battery", visual_state=0, seed=7).save(image_path)
        text = text or "It dies after two hours."
    aspects = aspects or predictor.aspect_vocab

    if not text or not image_path:
        parser.error("provide --text and --image, or use --demo")

    result = predictor.predict_sentiment(text, image_path, aspects)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
