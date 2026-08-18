import json
from pathlib import Path
from app.config import settings
from app.db import pending_training,mark_trained,training_status
def export_jsonl(rows):
    out=Path("data/training_pending.jsonl"); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"messages":[
              {"role":"system","content":"You are Satarka, an evidence-aware misinformation verification assistant."},
              {"role":"user","content":r["prompt"]},
              {"role":"assistant","content":r["completion"]}]},ensure_ascii=False)+"\n")
    return out
def train_if_ready(force=False):
    rows=pending_training()
    if not force and len(rows)<settings().training_batch_size:
        return {"status":"waiting","examples":len(rows),"message":f"Need {settings().training_batch_size-len(rows)} more verified examples."}
    if not rows:return {"status":"empty","examples":0,"message":"No training examples available."}
    if not settings().training_enabled:
        return {"status":"disabled","examples":len(rows),"message":"Training disabled. Set TRAINING_ENABLED=true on a suitable GPU machine."}
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments
        from trl import SFTTrainer
        from peft import LoraConfig
    except ImportError as e:
        return {"status":"missing_dependencies","examples":len(rows),"message":str(e)}
    path=export_jsonl(rows)
    ds=load_dataset("json",data_files=str(path),split="train")
    tok=AutoTokenizer.from_pretrained(settings().training_base_model)
    model=AutoModelForCausalLM.from_pretrained(settings().training_base_model)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    args=TrainingArguments(output_dir=settings().training_output_dir,num_train_epochs=1,
      per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=1e-4,
      logging_steps=5,save_strategy="epoch",report_to="none",fp16=False)
    peft=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias="none",task_type="CAUSAL_LM",
                    target_modules=["q_proj","k_proj","v_proj","o_proj"])
    trainer=SFTTrainer(model=model,args=args,train_dataset=ds,peft_config=peft)
    trainer.train(); trainer.save_model(settings().training_output_dir)
    mark_trained([r["id"] for r in rows])
    return {"status":"completed","examples":len(rows),"output_dir":settings().training_output_dir,"message":"LoRA adapter trained and saved."}
