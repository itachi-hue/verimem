"""
Compare activation steering on GPT-2 vs a small instruction-tuned model.

Run from repo root:
  python experiments/steering_os_commands.py

Requires: pip install torch transformers accelerate
"""

from __future__ import annotations

import argparse
import sys

import torch


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm() + 1e-8)


def gpt2_steering_experiment(scale: float, layer: int, greedy: bool) -> None:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    pos = "You are on Windows 11. PowerShell. List files in the current folder: "
    neg = "You are on Linux. bash. List files in the current folder: "

    # hidden_states[0]=embed; hidden_states[k+1]=after layer k — match hook on h[layer]
    hidx = layer + 1

    def last_hidden(prompt: str) -> torch.Tensor:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[hidx][:, -1, :]

    s = _normalize(last_hidden(pos) - last_hidden(neg))

    def steering_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + scale * s.to(hidden.device).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    prompt = "Generate a command to list files in the current directory."
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    def decode_new(outs):
        gen = outs[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    print("\n=== GPT-2 baseline (no steering, greedy) ===")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(decode_new(out))

    print(f"\n=== GPT-2 + steering (layer={layer}, scale={scale}, greedy={greedy}) ===")
    hook = model.transformer.h[layer].register_forward_hook(steering_hook)
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=not greedy,
                temperature=0.8 if not greedy else None,
                top_p=0.9 if not greedy else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(decode_new(out))
    finally:
        hook.remove()


def smollm_instruct_baseline_and_steering(scale: float, layer: int) -> None:
    """HuggingFaceTB/SmolLM2-360M-Instruct — chat template, middle-layer hook."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()

    # Contrast system messages (short, parallel)
    win_msgs = [
        {
            "role": "system",
            "content": "You are on Windows 11. Answer with PowerShell only, one line, no explanation.",
        },
        {"role": "user", "content": "Give a command to list files in the current directory."},
    ]
    nix_msgs = [
        {
            "role": "system",
            "content": "You are on Linux. Answer with bash only, one line, no explanation.",
        },
        {"role": "user", "content": "Give a command to list files in the current directory."},
    ]

    def encode_chat(msgs):
        return tokenizer.apply_chat_template(
            msgs, tokenize=True, return_tensors="pt", add_generation_prompt=False
        )

    hidx = layer + 1  # align with hook on layers[layer]

    def last_hidden_from_messages(msgs) -> torch.Tensor:
        ids = encode_chat(msgs)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        return out.hidden_states[hidx][:, -1, :]

    s = _normalize(last_hidden_from_messages(win_msgs) - last_hidden_from_messages(nix_msgs))

    user_only = [
        {"role": "user", "content": "List files in the current directory. One shell command only."}
    ]
    inputs = tokenizer.apply_chat_template(
        user_only, tokenize=True, return_tensors="pt", add_generation_prompt=True
    )

    def steering_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + scale * s.to(hidden.device).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    layers = model.model.layers
    prompt_len = inputs.shape[1]

    def decode_new(outs):
        gen = outs[0, prompt_len:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    print("\n=== SmolLM2-360M-Instruct baseline (no steering) ===")
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(decode_new(out))

    print(f"\n=== SmolLM2 + steering (layer={layer}, scale={scale}, greedy) ===")
    h = layers[layer].register_forward_hook(steering_hook)
    try:
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(decode_new(out))
    finally:
        h.remove()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gpt2-scale", type=float, default=0.6, help="Steering strength for GPT-2 (try 0.3–1.0)"
    )
    p.add_argument("--gpt2-layer", type=int, default=10)
    p.add_argument("--smolm-scale", type=float, default=0.8, help="Steering strength for SmolLM2")
    p.add_argument(
        "--smolm-layer",
        type=int,
        default=8,
        help="Decoder layer (0..N-1); SmolLM2 360M has ~16 layers",
    )
    p.add_argument(
        "--sample-gpt2", action="store_true", help="Use sampling on steered GPT-2 (often worse)"
    )
    args = p.parse_args()

    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    gpt2_steering_experiment(
        scale=args.gpt2_scale,
        layer=args.gpt2_layer,
        greedy=not args.sample_gpt2,
    )
    try:
        smollm_instruct_baseline_and_steering(scale=args.smolm_scale, layer=args.smolm_layer)
    except Exception as e:
        print(
            "\n[SmolLM2 section skipped or failed — install deps or check disk/network]",
            file=sys.stderr,
        )
        print(repr(e), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
