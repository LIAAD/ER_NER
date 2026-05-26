#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full fine-tune of MedAlBERTina (DeBERTa-v2) for Portuguese Clinical NER
with sliding-window chunking so long notes aren't truncated.

Targets (BIO):
  - Medicação Habitual                (no polarity)
  - Diagnóstico                       (no polarity)
  - Alergias medicamentosas__Positiva
  - Alergias medicamentosas__Negativa

Loader behavior (matches `final-dataset_complete.json` schema):
  * Only "Alergias medicamentosas" carries polarity. Any `Polaridade` field on
    "Medicação Habitual" or "Diagnóstico" annotations is silently ignored.
  * "Alergias medicamentosas" annotations without a valid Polaridade are dropped
    and counted in the [Load] line so schema drift is visible.
  * Sliding-window tokenization (long notes are chunked, not truncated).
  * Weighted cross-entropy: O capped at 0.25 of mean non-O weight; I-X tied to B-X
            so multi-token spans aren't truncated by the model.
  * Optional document-level oversampling for rare-polarity classes
            (--oversample_negativa N).
  * Per-class P/R/F1 logged at every validation epoch.
"""

import os, json, argparse, unicodedata, warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from dataset import Dataset, concatenate_datasets

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed,
    EarlyStoppingCallback
)
from transformers.trainer_utils import EvalPrediction
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# -------------------- Targets --------------------
# Per the current annotation schema, ONLY "Alergias medicamentosas" carries polarity.
# "Medicação Habitual" and "Diagnóstico" are flat (any Polaridade field is ignored).
POL_LABELS  = ["Alergias medicamentosas"]
FLAT_LABELS = ["Medicação Habitual", "Diagnóstico"]
POLARITIES  = ["Positiva", "Negativa"]

# Effective label space used in BIO tags:
#   Alergias medicamentosas__Positiva, Alergias medicamentosas__Negativa,
#   Medicação Habitual, Diagnóstico
COMBINED_LABELS = (
    [f"{lab}__{pol}" for lab in POL_LABELS for pol in POLARITIES]
    + FLAT_LABELS
)

# All base entity types (used for canonicalization)
ALL_BASE_LABELS = POL_LABELS + FLAT_LABELS

def _strip_accents_lower(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

# Canonicalizers
_CANON_BASE = {_strip_accents_lower(x): x for x in ALL_BASE_LABELS}
_CANON_POL  = {_strip_accents_lower(x): x for x in POLARITIES}

def canonical_base_label(raw: str) -> Optional[str]:
    return _CANON_BASE.get(_strip_accents_lower(raw or ""), None)

def canonical_polarity(raw: Optional[str]) -> Optional[str]:
    return _CANON_POL.get(_strip_accents_lower(raw or ""), None)

def resolve_overlaps_keep_longest(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    kept: List[List[object]] = []
    for b,e,l in spans:
        if not kept:
            kept.append([b,e,l]); continue
        kb,ke,kl = kept[-1]
        if b < ke:  # overlap
            if (e-b) > (ke-kb):
                kept[-1] = [b,e,l]
        else:
            kept.append([b,e,l])
    return [(int(b), int(e), str(l)) for b,e,l in kept]

def load_and_clean(path: str) -> List[Dict]:
    """Load annotations under the current 4-class schema:
       - Medicação Habitual                (no polarity; any Polaridade field is ignored)
       - Diagnóstico                       (no polarity; any Polaridade field is ignored)
       - Alergias medicamentosas__Positiva (polarity REQUIRED)
       - Alergias medicamentosas__Negativa (polarity REQUIRED)
    Annotations not matching any of these are counted in the [Load] line so
    schema drift is visible.
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    cleaned = []
    dropped_base = 0              # unknown/missing entity type
    dropped_alergias_no_pol = 0   # Alergias span with no polarity
    dropped_alergias_bad_pol = 0  # Alergias span with unrecognized polarity value
    ignored_pol_on_flat = 0       # Med. Habitual / Diag span with (ignored) polarity field
    kept = 0

    for ex in data:
        text = ex["text"]
        spans = []
        for a in ex.get("annotations", []):
            base = canonical_base_label(a.get("label"))
            if not base:
                dropped_base += 1
                continue

            b, e = int(a.get("begin", -1)), int(a.get("end", -1))
            if not (0 <= b < e <= len(text)):
                continue

            if base in POL_LABELS:
                # Alergias medicamentosas: polarity required
                pol = canonical_polarity(a.get("Polaridade"))
                if not pol:
                    raw_pol = a.get("Polaridade")
                    if raw_pol is None or str(raw_pol).strip() == "":
                        dropped_alergias_no_pol += 1
                    else:
                        dropped_alergias_bad_pol += 1
                    continue
                combined = f"{base}__{pol}"
            else:
                # Medicação Habitual / Diagnóstico: polarity (if any) is ignored.
                if a.get("Polaridade"):
                    ignored_pol_on_flat += 1
                combined = base

            spans.append((b, e, combined))

        spans = resolve_overlaps_keep_longest(spans)
        span_dicts = [{"begin": b, "end": e, "label": lab} for (b, e, lab) in spans]
        cleaned.append({"doc_id": ex.get("doc_id"), "text": text, "spans": span_dicts})
        kept += len(span_dicts)

    print(
        f"[Load] {path}\n"
        f"       kept spans: {kept} | "
        f"unknown entity type: {dropped_base} | "
        f"Alergias w/o polarity (dropped): {dropped_alergias_no_pol} | "
        f"Alergias w/ bad polarity (dropped): {dropped_alergias_bad_pol} | "
        f"polarity field ignored on flat label: {ignored_pol_on_flat}"
    )
    return cleaned

def build_label_list():
    tags = ["O"]
    for lab in COMBINED_LABELS:
        tags += [f"B-{lab}", f"I-{lab}"]
    return tags

# ---------- Sliding-window aligner ----------
def chunk_and_align_bio(example, tokenizer, label2id, max_len, stride):
    """
    Split a long note into overlapping chunks, project character spans per chunk.
    Returns a list of chunked rows to be flattened by Dataset.map with batched=True + remove_columns.
    """
    text = example["text"]
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_len,
        stride=stride,
        padding="max_length"   # pad each chunk for efficiency
    )

    all_rows = []
    n_chunks = len(enc["input_ids"])
    gold_spans = example.get("spans", [])
    covered_mentions = 0

    for ch in range(n_chunks):
        offsets = enc["offset_mapping"][ch]
        # chunk character range (excluding special tokens with (0,0))
        chunk_start = min([s for (s, e) in offsets if not (s == 0 and e == 0)] + [0])
        chunk_end   = max([e for (s, e) in offsets if not (s == 0 and e == 0)] + [0])

        labels = [label2id["O"]]*len(offsets)
        for i,(s,e) in enumerate(offsets):
            if s==e==0:
                labels[i] = -100  # special/pad

        # project only spans that intersect this chunk
        local_covered = 0
        for s in gold_spans:
            b = int(s["begin"]); e = int(s["end"]); lab = s["label"]
            if e <= chunk_start or b >= chunk_end:
                continue
            idxs = [i for i,(ts,te) in enumerate(offsets) if not (te<=b or e<=ts) and labels[i]!=-100]
            if not idxs:
                continue
            labels[idxs[0]] = label2id[f"B-{lab}"]
            for j in idxs[1:]:
                labels[j] = label2id[f"I-{lab}"]
            local_covered += 1

        covered_mentions += local_covered

        row = {k: enc[k][ch] for k in enc.keys() if k != "offset_mapping"}
        row["labels"] = labels
        all_rows.append(row)

    return {"chunks": all_rows, "covered_mentions": covered_mentions, "total_mentions": len(gold_spans)}

# ---- Per-class weights: inverse frequency; cap "O" relative to non-O mean
def compute_class_weights_from_dataset(ds, num_labels, o_index: int,
                                       id2label=None,
                                       min_w=0.10, max_w=10.0, o_cap_ratio=0.25,
                                       i_inherits_b=True):
    """Inverse-frequency class weights with two NER-specific adjustments:

    1. `o_cap_ratio` (default 0.25): caps the `O` weight at 25% of the
       mean non-O weight. Without this cap, `O` dominates the loss and
       the model under-predicts entities.

    2. `i_inherits_b`: after computing inverse-frequency weights, force every
       `I-X` weight to be at least as large as its corresponding `B-X` weight.
       Inverse-frequency systematically under-weights I- tokens (multi-token
       spans contribute many I-tokens but only one B-token), which causes the
       model to truncate spans early. Tying I- to B- protects span integrity.
       Requires `id2label` to identify B-/I- pairs by label name.
    """
    counts = np.zeros(num_labels, dtype=np.int64)
    for row in ds:
        for y in row["labels"]:
            if y != -100:
                counts[y] += 1
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts.astype(np.float32)
    weights = inv / inv.mean()
    non_o = [i for i in range(num_labels) if i != o_index]
    mean_non_o = float(np.mean([weights[i] for i in non_o])) if non_o else 1.0
    weights[o_index] = min(float(weights[o_index]), mean_non_o * o_cap_ratio)

    # Tie I-X weight to B-X weight (use the larger of the two for both)
    if i_inherits_b and id2label is not None:
        label2id_local = {lbl: i for i, lbl in id2label.items()}
        for lbl, i in label2id_local.items():
            if not lbl.startswith("B-"):
                continue
            i_lbl = "I-" + lbl[2:]
            j = label2id_local.get(i_lbl)
            if j is None:
                continue
            shared = max(float(weights[i]), float(weights[j]))
            weights[i] = shared
            weights[j] = shared

    weights = np.clip(weights, min_w, max_w)
    return torch.tensor(weights)

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            ignore_index=-100
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def _save_optimizer_and_scheduler(self, output_dir: str):
        return

def compute_metrics_fn(id2label):
    def _to_tags(pred_ids, label_ids):
        tags_pred, tags_true = [], []
        for p, y in zip(pred_ids, label_ids):
            cur_p, cur_y = [], []
            for pi, yi in zip(p, y):
                if yi == -100:
                    continue
                cur_p.append(id2label[int(pi)])
                cur_y.append(id2label[int(yi)])
            tags_pred.append(cur_p); tags_true.append(cur_y)
        return tags_true, tags_pred

    def _metrics(eval_pred: EvalPrediction):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids
        y_true, y_pred = _to_tags(preds, labels)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tk_true = [t for seq in y_true for t in seq]
        tk_pred = [t for seq in y_pred for t in seq]
        token_acc = np.mean([a==b for a,b in zip(tk_true, tk_pred)]) if tk_true else 0.0
        out = {"precision": p, "recall": r, "f1": f1, "token_acc": token_acc}

        # Per-class F1 (4 entity classes): seqeval's classification_report
        # reports BIO-stripped class names (e.g. "Alergias medicamentosas__Negativa",
        # "Medicação Habitual", "Diagnóstico").
        # Surface these as scalar metrics so they appear in trainer logs each epoch.
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            for cls_name, scores in report.items():
                if cls_name in {"micro avg", "macro avg", "weighted avg", "accuracy"}:
                    continue
                if not isinstance(scores, dict):
                    continue
                # Compact key: replace problematic chars so the trainer's logger handles them.
                key = cls_name.replace(" ", "_").replace("__", "_")
                out[f"f1_{key}"] = float(scores.get("f1-score", 0.0))
                out[f"p_{key}"]  = float(scores.get("precision", 0.0))
                out[f"r_{key}"]  = float(scores.get("recall", 0.0))
                out[f"n_{key}"]  = int(scores.get("support", 0))
        except Exception as e:
            # Don't let a metric-formatting bug crash training.
            print(f"[compute_metrics] per-class report failed: {e}")
        return out
    return _metrics

def flatten_chunk_map(batch):
    # batch["chunks"] is a list of lists of chunk dicts; flatten to a new batch
    out = {k: [] for k in ["input_ids","token_type_ids","attention_mask","labels"]}
    for chunks in batch["chunks"]:
        for ch in chunks:
            for k in out.keys():
                if k in ch:
                    out[k].append(ch[k])
                else:
                    # token_type_ids may be missing for some models
                    if k == "token_type_ids":
                        out[k].append(None)
    # remove None lists if entirely None
    if all(x is None for x in out["token_type_ids"]):
        out.pop("token_type_ids")
    return out

def oversample_minority_polarity(
    docs: List[Dict],
    target_label: str,
    target_polarity: str,
    factor: int,
) -> List[Dict]:
    """Duplicate documents that contain at least one span of (target_label, target_polarity).
    Operates at document level (not chunk level) so each replicated doc still gets its
    own sliding-window chunking; gradient steps then see the same negation context
    multiple times per epoch. No-op when factor <= 1.

    Returns a new list (input is not mutated). Each duplicated dict is a shallow copy
    with `spans` deep-copied; this is enough because downstream chunking only reads
    text + spans and assigns its own input ids/labels per chunk.
    """
    if factor <= 1:
        return docs
    target_combined = f"{target_label}__{target_polarity}"
    out = []
    n_seed_docs = 0
    for d in docs:
        out.append(d)
        has_target = any(s.get("label") == target_combined for s in d.get("spans", []))
        if has_target:
            n_seed_docs += 1
            for _ in range(factor - 1):
                out.append({
                    "doc_id": d.get("doc_id"),
                    "text":   d.get("text"),
                    "spans":  [dict(s) for s in d.get("spans", [])],
                })
    print(
        f"[Oversample] base docs: {len(docs)} | "
        f"docs with {target_combined}: {n_seed_docs} | "
        f"factor: x{factor} | "
        f"final docs: {len(out)} (added {len(out) - len(docs)})"
    )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", type=str, required=True)
    ap.add_argument("--val_json",   type=str, required=True)
    ap.add_argument("--model", type=str, default="portugueseNLP/medialbertina_pt-pt_900m")
    ap.add_argument("--output_dir", type=str, default="outputs_fullfinetune_chunked")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--precision", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--oversample_negativa", type=int, default=1,
                    help="Duplicate each training doc containing an Alergias medicamentosas__Negativa span "
                         "this many times (1 = no oversampling, 4-5 recommended for low-resource Negativa).")
    args = ap.parse_args()

    if args.gpu is not None and str(args.gpu).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()

    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_fp16 = False
    bf16_ok = (args.precision == "bf16") and (device_count > 0)
    fp16_ok = (args.precision == "fp16") and (device_count > 0)
    if bf16_ok:
        torch_dtype = torch.bfloat16
        os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
        print("[Precision] Using bf16.")
    elif fp16_ok:
        torch_dtype = torch.float16
        os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
        use_fp16 = True
        print("[Precision] Using fp16.")
    else:
        torch_dtype = torch.float32
        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
        use_fp16 = False
        print("[Precision] Using fp32.")

    # ----- Data -----
    data_train = load_and_clean(args.train_json)
    data_val   = load_and_clean(args.val_json)

    # Oversample documents containing rare-polarity spans (training set only).
    # Boosts gradient signal for `Alergias medicamentosas__Negativa` (~82 train spans
    # in ~30 docs vs. >1100 Medicação Habitual spans). Validation stays untouched
    # to keep the val metric a fair early-stopping signal.
    data_train = oversample_minority_polarity(
        data_train,
        target_label="Alergias medicamentosas",
        target_polarity="Negativa",
        factor=args.oversample_negativa,
    )

    if len(data_train) < 2: raise ValueError("Train set too small after cleaning.")
    if len(data_val)   < 2: raise ValueError("Validation set too small after cleaning.")

    label_list = build_label_list()
    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}

    print(f"\n[Label space] {len(label_list)} labels (1 'O' + {len(COMBINED_LABELS)} entity classes × B/I):")
    for i, lbl in enumerate(label_list):
        print(f"  {i:2d}  {lbl}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def chunk_map_fn(ex):
        return chunk_and_align_bio(ex, tokenizer, label2id, max_len=args.max_len, stride=args.stride)

    ds_train_raw = Dataset.from_list(data_train).map(chunk_map_fn, batched=False)
    ds_val_raw   = Dataset.from_list(data_val).map(chunk_map_fn, batched=False)

    # Coverage sanity check
    cov_tr = sum(ds_train_raw["covered_mentions"])
    tot_tr = sum(ds_train_raw["total_mentions"])
    cov_va = sum(ds_val_raw["covered_mentions"])
    tot_va = sum(ds_val_raw["total_mentions"])
    print(f"[Coverage] Train covered {cov_tr}/{tot_tr} mentions ({100.0*cov_tr/max(1,tot_tr):.1f}%).")
    print(f"[Coverage]  Val  covered {cov_va}/{tot_va} mentions ({100.0*cov_va/max(1,tot_va):.1f}%).")

    # Flatten chunks into token-classification rows
    ds_train = ds_train_raw.map(flatten_chunk_map, batched=True, remove_columns=ds_train_raw.column_names)
    ds_val   = ds_val_raw.map(flatten_chunk_map, batched=True, remove_columns=ds_val_raw.column_names)

    # ----- Model -----
    config = AutoConfig.from_pretrained(
        args.model, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, config=config, torch_dtype=torch_dtype
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metrics_fn = compute_metrics_fn(id2label)

    # class weights (per-class inverse freq, cap 'O', tie I- to B- to protect multi-token spans)
    o_index = label2id["O"]
    class_w = compute_class_weights_from_dataset(
        ds_train,
        num_labels=len(label_list),
        o_index=o_index,
        id2label=id2label,
    )
    print("[Class Weights] (index -> weight)")
    for i,w in enumerate(class_w.tolist()):
        print(f"  {i:2d} {id2label[i]:40s} : {w:.3f}")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",   
        save_strategy="epoch",
        save_total_limit=2,            
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=0.10,
        lr_scheduler_type="cosine",
        fp16=use_fp16,
        bf16=bf16_ok,
        label_smoothing_factor=0.0,   
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        logging_steps=25,
        seed=42,
        save_safetensors=True,
        overwrite_output_dir=True,
        # save_only_model=True,  # removed: requires transformers>=4.36 (we pin 4.35.2)
    )

    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.patience, early_stopping_threshold=0.0
        ))

    trainer = WeightedLossTrainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer,                     # transformers==4.35.2 uses `tokenizer=`, not `processing_class=`
        compute_metrics=metrics_fn,
        class_weights=class_w,
        callbacks=callbacks,
    )

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\n[CUDA OOM] Lower --batch_size or --max_len, or increase --grad_accum.\n")
        raise

    # ----- Save best & report -----
    best_dir = os.path.join(args.output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    eval_res = trainer.evaluate()
    print("\n=== Validation Metrics ===")
    for k,v in eval_res.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")

    preds = trainer.predict(ds_val)
    y_true, y_pred = [], []
    for p, lab in zip(np.argmax(preds.predictions, axis=-1), preds.label_ids):
        t_true, t_pred = [], []
        for pi, yi in zip(p, lab):
            if yi == -100: continue
            t_true.append(id2label[int(yi)])
            t_pred.append(id2label[int(pi)])
        y_true.append(t_true); y_pred.append(t_pred)
    print("\nSeqEval report:\n", classification_report(y_true, y_pred))
    print("\n[Note] Per-class numbers are also logged at every epoch under keys like "
          "'eval_f1_Alergias_medicamentosas_Negativa', 'eval_f1_Medicação_Habitual', etc.")

if __name__ == "__main__":
    main()