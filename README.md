# Shakespeare Byte-Level Transformer

A small Transformer language model trained from scratch on the complete works of William Shakespeare, using byte-level tokenization (vocab size 257).

**Architecture:** 6-layer decoder-only Transformer, 256 embedding dim, 8 attention heads, ~4.9M parameters.

**Training:** 20,000 steps on ~5.4M bytes of text, final train loss ~1.21.

## Files

- `model.py` — model architecture
- `app.py` — Gradio web app for text generation
- `shakespeare_model_inference.pt` — trained weights
- `requirements.txt` — dependencies

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints in your terminal.

## Example

Prompt: `ROMEO:` → generates Shakespeare-style dialogue, stage directions, and verse.
