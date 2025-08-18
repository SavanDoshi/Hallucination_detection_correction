
import argparse, json, re
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompts import build_prompt
from .eval_metrics import best_em_f1

def apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Use tokenizer's chat template when available; otherwise fall back."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text
    except Exception:
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

def load_df(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="outputs.jsonl")
    ap.add_argument("--metrics", default="metrics.json")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"[Loading model: {args.model}]")
    quant_kwargs = {}
    if args.load_in_4bit:
        quant_kwargs.update(dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            device_map="auto",
        ))
        torch_dtype = None
    else:
        quant_kwargs.update(dict(device_map="auto"))
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **quant_kwargs
    )
    model.eval()

    df = load_df(args.input)

    preds = []
    em_vec, f1_vec = [], []
    for _, ex in df.iterrows():
        qid = ex.get("id")
        q = ex.get("question")
        ctx = ex.get("context", None)
        answers = ex.get("answers", None)
        if answers is None and pd.notna(ex.get("answer", None)):
            answers = [ex.get("answer")]
        choices = ex.get("choices", None)

        sys_prompt, user_prompt = build_prompt(q, ctx, choices)
        prompt_text = apply_chat_template(tok, sys_prompt, user_prompt)

        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                pad_token_id=tok.eos_token_id,
            )
        out_text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if choices is not None:
            m = re.search(r"\b([A-J])\b", out_text)
            if m:
                out_text = m.group(1)

        pred_obj = {"id": qid, "prediction": out_text}
        if answers is not None:
            em, f1 = best_em_f1(out_text, answers)
            pred_obj["em"], pred_obj["f1"] = em, f1
            em_vec.append(em); f1_vec.append(f1)
        preds.append(pred_obj)

    with open(args.output, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    metrics = {}
    if em_vec:
        metrics = {
            "N": int(len(em_vec)),
            "EM": float(accuracy_score([1]*len(em_vec), [1 if x==1.0 else 0 for x in em_vec])),
            "F1": float(sum(f1_vec)/len(f1_vec))
        }
        with open(args.metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print("[Done]")
    if metrics:
        print(f"EM: {metrics['EM']:.3f}  F1: {metrics['F1']:.3f}  (N={metrics['N']})")
    else:
        print("No gold answers provided; only wrote predictions.")

if __name__ == "__main__":
    main()
