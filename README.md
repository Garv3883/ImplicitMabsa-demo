# ImplicitMABSA: 3-Stream Multimodal Aspect-Based Sentiment Analysis

A deep learning framework and real-time **FastAPI** inference demo for **Multimodal Aspect-Based Sentiment Analysis (MABSA)** on social media posts (text + image pairs), trained and evaluated on the **Twitter-2015** benchmark.

---

## Architecture Overview

To isolate unimodal bias from true cross-modal reasoning, **ImplicitMABSA** implements a unified 3-stream architecture:

1. **Stream A — Text-Only Baseline (`DeBERTa-v3`)**: Encodes `[CLS] tweet [SEP] aspect [SEP]` pairs to predict aspect-level sentiment polarity and aspect-text alignment.
2. **Stream B — Image-Only Baseline (`ViT`)**: Encodes accompanying post images via Vision Transformer patch embeddings to capture pure visual sentiment bias without text cues.
3. **Stream C — Multimodal Cross-Attention Fusion**: Aligns DeBERTa-v3 token representations with ViT visual patch embeddings using multi-head cross-modal attention to resolve implicit aspects and mixed-sentiment posts.

---

## Benchmark Results (Twitter-2015 Test Set)

| Stream | Modality | Test Accuracy | Aspect Macro-F1 |
| :--- | :--- | :---: | :---: |
| **Stream A** | Text-Only (DeBERTa-v3) | — | — |
| **Stream B** | Image-Only (ViT) | — | `n/a` |
| **Stream C** | **Multimodal Fusion** | **79.78%** | **0.9963** |

*(Model weights hosted on Hugging Face Hub: [`Garv3883/implicit-mabsa-weights`](https://huggingface.co/Garv3883/implicit-mabsa-weights))*

---

## Project Structure

```text
├── app.py              # FastAPI server & interactive live inference UI
├── model.py            # 3-stream DeBERTa-v3 + ViT cross-modal attention model
├── dataset.py          # Twitter-2015 preprocessing & candidate aspect expansion
├── train.py            # Multi-stream training pipeline
├── evaluate.py         # Ablation evaluation & benchmark metrics
├── inference.py        # CLI inference helper
├── prepare_data.py     # Dataset preparation utilities
├── utils.py            # Shared helper functions
├── Dockerfile          # Container setup for Hugging Face Spaces deployment
└── requirements.txt    # Python dependencies
