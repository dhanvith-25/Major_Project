import json
from pathlib import Path

from app.config import settings
from app.db import mark_trained, pending_training, training_status


def export_jsonl(rows):
    out = Path("data/training_pending.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "You are Satarka, an evidence-aware misinformation verification assistant."},
                            {"role": "user", "content": r["prompt"]},
                            {"role": "assistant", "content": r["completion"]},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out


def _build_quantized_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _evaluate_model(model, tokenizer, eval_rows):
    try:
        from nltk.translate.bleu_score import corpus_bleu
        from rouge_score import rouge_scorer
    except ImportError as exc:
        return {"status": "missing_eval_deps", "message": str(exc), "rouge_l": 0.0, "rouge_2": 0.0, "bleu": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = []
    bleu_scores = []

    for row in eval_rows:
        prompt = row["prompt"]
        expected = row["completion"]
        generated = model.generate(
            tokenizer(prompt, return_tensors="pt").to(model.device),
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        if "\n" in decoded:
            decoded = decoded.split("\n")[-1].strip()
        rouge_scores.append(scorer.score(expected, decoded))
        bleu_scores.append(corpus_bleu([[expected.split()]], [decoded.split()]))

    avg_rouge_l = sum(score["rougeL"].fmeasure for score in rouge_scores) / max(1, len(rouge_scores))
    avg_rouge_2 = sum(score["rouge2"].fmeasure for score in rouge_scores) / max(1, len(rouge_scores))
    avg_bleu = sum(bleu_scores) / max(1, len(bleu_scores))
    return {"rouge_l": round(avg_rouge_l, 4), "rouge_2": round(avg_rouge_2, 4), "bleu": round(avg_bleu, 4)}


def train_if_ready(force=False):
    rows = pending_training()
    if not force and len(rows) < settings().training_batch_size:
        return {"status": "waiting", "examples": len(rows), "message": f"Need {settings().training_batch_size - len(rows)} more verified examples."}
    if not rows:
        return {"status": "empty", "examples": 0, "message": "No training examples available."}
    if not settings().training_enabled:
        return {"status": "disabled", "examples": len(rows), "message": "Training disabled. Set TRAINING_ENABLED=true on a suitable GPU machine."}

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import TrainingArguments
        from trl import SFTTrainer
    except ImportError as e:
        return {"status": "missing_dependencies", "examples": len(rows), "message": str(e)}

    path = export_jsonl(rows)
    ds = load_dataset("json", data_files=str(path), split="train")

    model, tokenizer = _build_quantized_model(settings().training_base_model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        output_dir=settings().training_output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        fp16=False,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    metrics = _evaluate_model(model, tokenizer, rows)
    final_dir = settings().training_output_dir
    trainer.save_model(final_dir)
    mark_trained([r["id"] for r in rows])

    return {
        "status": "completed",
        "examples": len(rows),
        "output_dir": final_dir,
        "message": "LoRA adapter trained and saved.",
        "metrics": metrics,
    }
