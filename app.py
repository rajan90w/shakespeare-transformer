import os
import torch
import torch.nn.functional as F
import gradio as gr

from model import ByteLevelTransformer

DEVICE = "cpu"  # Render free tier is CPU-only
CHECKPOINT_PATH = "shakespeare_model_inference.pt"

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = ByteLevelTransformer(
    vocab_size=257,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    ff_dim=1024,
    max_seq_len=256,
    dropout=0.1
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

trained_steps = checkpoint.get("step", "unknown")
trained_loss = checkpoint.get("train_loss", "unknown")


# --------------------------------------------------
# Generation function (same logic as your Kaggle notebook)
# --------------------------------------------------

@torch.no_grad()
def generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    if not prompt:
        prompt = " "

    prompt_bytes = prompt.encode("utf-8")
    tokens = [b + 1 for b in prompt_bytes]
    x = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(int(max_new_tokens)):
        x_cond = x[:, -256:]
        logits = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, int(top_k))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

    out_tokens = x[0].tolist()
    out_bytes = bytes([t - 1 for t in out_tokens if 1 <= t <= 256])
    return out_bytes.decode("utf-8", errors="replace")


# --------------------------------------------------
# Gradio UI
# --------------------------------------------------

loss_display = f"{trained_loss:.4f}" if isinstance(trained_loss, float) else str(trained_loss)

description = f"""
A byte-level Transformer trained from scratch on the complete works of
William Shakespeare. Trained for {trained_steps} steps
(final train loss: {loss_display}).
Enter a prompt (e.g. "ROMEO:", "HAMLET:", "To be or not to be") and generate text.
"""

with gr.Blocks(title="Shakespeare Byte-Level Transformer") as demo:
    gr.Markdown("# 🖋️ Shakespeare Byte-Level Transformer")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            prompt_box = gr.Textbox(
                label="Prompt",
                value="ROMEO:",
                placeholder="Type a prompt, e.g. ROMEO:"
            )
            max_tokens_slider = gr.Slider(
                minimum=20, maximum=500, value=200, step=10,
                label="Max new tokens (bytes)"
            )
            temperature_slider = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.8, step=0.05,
                label="Temperature (higher = more random)"
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=200, value=40, step=1,
                label="Top-k sampling"
            )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_box = gr.Textbox(
                label="Generated text",
                lines=15
            )

    generate_btn.click(
        fn=generate,
        inputs=[prompt_box, max_tokens_slider, temperature_slider, top_k_slider],
        outputs=output_box
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
