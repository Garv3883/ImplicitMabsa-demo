"""Unified model for the ImplicitMABSA comparative ablation study.

Streams
-------
* ``text_only``  (Stream A) DeBERTa-v3 over ``[CLS] text [SEP] candidate aspect [SEP]``
* ``image_only`` (Stream B) ViT over the image (new classification head on the ViT [CLS] token)
* ``multimodal`` (Stream C) text tokens (queries) cross-attend to ViT patch tokens
                 (keys/values) via ``nn.MultiheadAttention`` -> MLP head

Every stream predicts polarity (3 classes). Streams that see the candidate aspect
(A and C) additionally predict *aspect relevance* (is the candidate the target
implicit aspect? 2 classes). The image-only stream has no access to the candidate
text, so it has no aspect head.

Streams are meant to be trained independently: build one model per stream with
``ImplicitMABSAModel(cfg, modes=(mode,))`` so each owns its own encoder copies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, ViTConfig, ViTModel

VALID_MODES: Tuple[str, ...] = ("text_only", "image_only", "multimodal")
STREAM_DISPLAY_NAMES: Dict[str, str] = {
    "text_only": "Stream A (Text-Only)",
    "image_only": "Stream B (Image-Only)",
    "multimodal": "Stream C (Multimodal)",
}
# Parameter-name prefixes owned by each stream (used to save/load per-stream weights).
STREAM_PARAM_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "text_only": ("text_encoder.", "text_sentiment_head.", "text_aspect_head."),
    "image_only": ("vision_encoder.", "vision_sentiment_head."),
    "multimodal": (
        "text_encoder.", "vision_encoder.", "cross_attention.",
        "multimodal_sentiment_head.", "multimodal_aspect_head.",
    ),
}


@dataclass
class ModelConfig:
    """Hyper-parameters needed to rebuild the architecture (stored in the checkpoint)."""

    text_model_name: str = "microsoft/deberta-v3-base"  # or "bert-base-uncased"
    vision_model_name: str = "google/vit-base-patch16-224"
    num_sentiments: int = 3
    num_aspect_classes: int = 2
    fusion_heads: int = 8
    head_hidden_dim: int = 512
    dropout: float = 0.1
    freeze_text: bool = False
    freeze_vision: bool = False


class MLPHead(nn.Module):
    """LayerNorm -> Dropout -> Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalFusion(nn.Module):
    """Text-to-image cross-modal attention block.

    Text tokens are the queries; projected ViT patch tokens are keys and values, so
    every word can pull in the visual evidence that resolves an omitted aspect.
    A residual connection, LayerNorm and a feed-forward sub-layer follow (Transformer style).
    """

    def __init__(self, text_dim: int, vision_dim: int, num_heads: int, dropout: float, ffn_mult: int = 4) -> None:
        super().__init__()
        self.visual_proj = nn.Identity() if vision_dim == text_dim else nn.Linear(vision_dim, text_dim)
        self.attn = nn.MultiheadAttention(text_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(text_dim)
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, ffn_mult * text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * text_dim, text_dim),
        )
        self.norm2 = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, text_hidden: torch.Tensor, image_hidden: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns (fused text tokens [B, L, D], projected image tokens [B, P, D], attn weights or None)."""
        visual = self.visual_proj(image_hidden)
        attn_out, attn_weights = self.attn(
            query=text_hidden, key=visual, value=visual,
            need_weights=return_attention, average_attn_weights=True,
        )
        x = self.norm1(text_hidden + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, visual, attn_weights


class ImplicitMABSAModel(nn.Module):
    """Three-stream model: text-only, image-only and cross-attention multimodal fusion.

    Args:
        config: architecture hyper-parameters.
        modes: which streams to instantiate (default: all three). Components that no
            requested stream needs are not built, which saves memory.
        pretrained: load pre-trained encoder weights. Use ``False`` when you are about
            to restore fine-tuned weights anyway (inference / evaluation).

    Note: if several modes are instantiated together, ``text_only`` and ``multimodal``
    share one ``text_encoder``. For independent ablations build one model per mode.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        modes: Sequence[str] = VALID_MODES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.modes: Tuple[str, ...] = tuple(modes)
        for m in self.modes:
            self._validate_mode(m)
        cfg = self.config

        need_text = any(m in self.modes for m in ("text_only", "multimodal"))
        need_vision = any(m in self.modes for m in ("image_only", "multimodal"))

        text_dim = vision_dim = 0
        if need_text:
            self.text_encoder = self._build_text_encoder(cfg.text_model_name, pretrained)
            text_dim = int(self.text_encoder.config.hidden_size)
            if cfg.freeze_text:
                self._freeze(self.text_encoder)
        if need_vision:
            self.vision_encoder = self._build_vision_encoder(cfg.vision_model_name, pretrained)
            vision_dim = int(self.vision_encoder.config.hidden_size)
            if cfg.freeze_vision:
                self._freeze(self.vision_encoder)

        def head(in_dim: int, out_dim: int) -> MLPHead:
            return MLPHead(in_dim, cfg.head_hidden_dim, out_dim, cfg.dropout)

        if "text_only" in self.modes:
            self.text_sentiment_head = head(text_dim, cfg.num_sentiments)
            self.text_aspect_head = head(text_dim, cfg.num_aspect_classes)
        if "image_only" in self.modes:
            self.vision_sentiment_head = head(vision_dim, cfg.num_sentiments)
        if "multimodal" in self.modes:
            self.cross_attention = CrossModalFusion(text_dim, vision_dim, cfg.fusion_heads, cfg.dropout)
            fused_dim = 3 * text_dim  # [fused CLS | text CLS | image CLS]
            self.multimodal_sentiment_head = head(fused_dim, cfg.num_sentiments)
            self.multimodal_aspect_head = head(fused_dim, cfg.num_aspect_classes)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_mode(mode: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {VALID_MODES}.")

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def _build_text_encoder(name: str, pretrained: bool) -> nn.Module:
        if pretrained:
            return AutoModel.from_pretrained(name, torch_dtype=torch.float32)
        return AutoModel.from_config(AutoConfig.from_pretrained(name))

    @staticmethod
    def _build_vision_encoder(name: str, pretrained: bool) -> ViTModel:
        if pretrained:
            return ViTModel.from_pretrained(name, add_pooling_layer=False, torch_dtype=torch.float32)
        return ViTModel(ViTConfig.from_pretrained(name), add_pooling_layer=False)

    # ------------------------------------------------------------------ #
    # Encoders
    # ------------------------------------------------------------------ #
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Token-level text features [B, L, D]."""
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # DeBERTa-v3 has type_vocab_size == 0 and must not receive token_type_ids; BERT needs them.
        if token_type_ids is not None and getattr(self.text_encoder.config, "type_vocab_size", 0) > 0:
            kwargs["token_type_ids"] = token_type_ids
        return self.text_encoder(**kwargs).last_hidden_state

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Patch-level visual features [B, 1 + num_patches, D] (index 0 is the ViT [CLS] token)."""
        return self.vision_encoder(pixel_values=pixel_values).last_hidden_state

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        mode: str = "multimodal",
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Run one stream.

        Args:
            mode: ``'text_only'``, ``'image_only'`` or ``'multimodal'``.
            input_ids / attention_mask / token_type_ids: tokenised ``[CLS] text [SEP] aspect [SEP]``.
            pixel_values: preprocessed images [B, 3, 224, 224].
            return_attention: also return text->patch attention [B, L, P] (multimodal only).

        Returns:
            dict with ``sentiment_logits`` [B, 3], ``aspect_logits`` [B, 2] (``None`` for
            image-only) and, optionally, ``attention_weights``.
        """
        self._validate_mode(mode)
        if mode not in self.modes:
            raise ValueError(f"Mode {mode!r} was not built; this model contains {self.modes}.")

        if mode in ("text_only", "multimodal") and (input_ids is None or attention_mask is None):
            raise ValueError(f"mode={mode!r} requires input_ids and attention_mask.")
        if mode in ("image_only", "multimodal") and pixel_values is None:
            raise ValueError(f"mode={mode!r} requires pixel_values.")

        if mode == "text_only":
            text_hidden = self.encode_text(input_ids, attention_mask, token_type_ids)
            pooled = text_hidden[:, 0]
            return {
                "sentiment_logits": self.text_sentiment_head(pooled),
                "aspect_logits": self.text_aspect_head(pooled),
            }

        if mode == "image_only":
            image_hidden = self.encode_image(pixel_values)
            return {
                "sentiment_logits": self.vision_sentiment_head(image_hidden[:, 0]),
                "aspect_logits": None,
            }

        text_hidden = self.encode_text(input_ids, attention_mask, token_type_ids)
        image_hidden = self.encode_image(pixel_values)
        fused, visual, attn = self.cross_attention(text_hidden, image_hidden, return_attention)
        features = torch.cat([fused[:, 0], text_hidden[:, 0], visual[:, 0]], dim=-1)
        out: Dict[str, Optional[torch.Tensor]] = {
            "sentiment_logits": self.multimodal_sentiment_head(features),
            "aspect_logits": self.multimodal_aspect_head(features),
        }
        if return_attention:
            out["attention_weights"] = attn
        return out

    # ------------------------------------------------------------------ #
    # Optimiser groups & per-stream (de)serialisation
    # ------------------------------------------------------------------ #
    def parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float) -> list:
        """AdamW groups: pre-trained encoders vs. new layers; no weight decay for biases / norms."""
        buckets: Dict[Tuple[bool, bool], list] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_encoder = name.startswith(("text_encoder.", "vision_encoder."))
            decay = param.ndim > 1
            buckets.setdefault((is_encoder, decay), []).append(param)
        return [
            {
                "params": params,
                "lr": encoder_lr if is_encoder else head_lr,
                "weight_decay": weight_decay if decay else 0.0,
            }
            for (is_encoder, decay), params in buckets.items()
        ]

    def stream_state_dict(self, mode: str) -> Dict[str, torch.Tensor]:
        """CPU copy of only the weights that ``mode`` uses."""
        self._validate_mode(mode)
        prefixes = STREAM_PARAM_PREFIXES[mode]
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith(prefixes)
        }

    def load_stream_state_dict(self, mode: str, state_dict: Dict[str, torch.Tensor]) -> None:
        """Restore weights saved by :meth:`stream_state_dict`; raises if anything is missing."""
        self._validate_mode(mode)
        prefixes = STREAM_PARAM_PREFIXES[mode]
        result = self.load_state_dict(state_dict, strict=False)
        missing = [k for k in result.missing_keys if k.startswith(prefixes)]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"State dict mismatch for stream {mode!r}: missing={missing[:5]}, "
                f"unexpected={list(result.unexpected_keys)[:5]}"
            )

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def select_inputs(batch: Dict[str, torch.Tensor], mode: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Move only the tensors that ``mode`` needs onto ``device`` (keyword args for ``forward``)."""
    inputs: Dict[str, torch.Tensor] = {}
    if mode in ("text_only", "multimodal"):
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in batch:
                inputs[key] = batch[key].to(device, non_blocking=True)
    if mode in ("image_only", "multimodal"):
        inputs["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True)
    return inputs
