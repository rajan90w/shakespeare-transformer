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
# Generation function (KV-cached, much faster than recomputing
# the full sequence through all layers on every step)
# --------------------------------------------------

NUM_HEADS = model.layers[0].self_attention.num_heads
EMBED_DIM = model.token_embedding.embedding_dim
HEAD_DIM = EMBED_DIM // NUM_HEADS


@torch.no_grad()
def generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    if not prompt:
        prompt = " "

    max_new_tokens = int(max_new_tokens)
    top_k = int(top_k) if top_k else None

    prompt_bytes = prompt.encode("utf-8")
    tokens = [b + 1 for b in prompt_bytes]
    if not tokens:
        tokens = [1]
    # keep room for generated tokens within max_seq_len
    tokens = tokens[-(model.max_seq_len - 1):]

    k_cache = [None] * len(model.layers)
    v_cache = [None] * len(model.layers)

    def forward_step(token_id, pos):
        x = (
            model.token_embedding(torch.tensor([[token_id]], device=DEVICE))
            + model.position_embedding(torch.tensor([[pos]], device=DEVICE))
        )  # [1, 1, embed_dim]

        for i, layer in enumerate(model.layers):
            attn = layer.self_attention
            qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            def reshape(t):
                return t.view(1, 1, NUM_HEADS, HEAD_DIM).transpose(1, 2)

            q, k, v = reshape(q), reshape(k), reshape(v)

            if k_cache[i] is None:
                k_cache[i], v_cache[i] = k, v
            else:
                k_cache[i] = torch.cat([k_cache[i], k], dim=2)
                v_cache[i] = torch.cat([v_cache[i], v], dim=2)

            scale = HEAD_DIM ** -0.5
            scores = torch.matmul(q, k_cache[i].transpose(-2, -1)) * scale
            probs = F.softmax(scores, dim=-1)
            out = torch.matmul(probs, v_cache[i])
            out = out.transpose(1, 2).reshape(1, 1, EMBED_DIM)
            out = attn.out_proj(out)

            x = layer.norm1(x + layer.dropout(out))
            ff_out = layer.feed_forward(x)
            x = layer.norm2(x + layer.dropout(ff_out))

        x = model.layer_norm(x)
        logits = model.output_projection(x)
        return logits[:, -1, :]

    logits = None
    for pos, tok in enumerate(tokens):
        logits = forward_step(tok, pos)

    generated = list(tokens)
    pos = len(tokens)

    for _ in range(max_new_tokens):
        if pos >= model.max_seq_len:
            break

        step_logits = logits / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            v_top, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
            step_logits = step_logits.clone()
            step_logits[step_logits < v_top[:, [-1]]] = -float("inf")

        probs = F.softmax(step_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)

        logits = forward_step(next_token, pos)
        pos += 1

    out_bytes = bytes([t - 1 for t in generated if 1 <= t <= 256])
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
