#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Polarity evaluator for Portuguese Clinical NER.

Targets (4 classes, matching train_sub_alldis.py schema):
  - Medicação Habitual                (no polarity — flat label)
  - Diagnóstico                       (no polarity — flat label)
  - Alergias medicamentosas__Positiva (polarity required)
  - Alergias medicamentosas__Negativa (polarity required)

What it does
------------
- Loads a token classification model (auto-picks `best/` if present).
- Runs sliding-window inference so long documents are not truncated.
- Decodes predicted spans and extracts (base, polarity?) from tag strings.
- EXACT match: base must match AND (begin,end) identical to gold.
- RELAXED IoU@T match: base must match AND IoU(predicted,gold) >= threshold.
- Two scoring modes:
    * matched — evaluate polarity only on matched spans
    * joint   — evaluate extraction + polarity (penalize missed/spurious spans)
- Reports both polarity-level (Positiva/Negativa) AND per-BIO+polarity-class
  (all 4 classes including Medicação Habitual and Diagnóstico).

Example
-------
python test-per-class.py \
  --model_dir outputs_fullfinetune_balanced/best \
  --test_json final-data/test_balanced.json \
  --max_len 512 --stride 128 \
  --score_mode joint \
  --pred_json polarity_eval.json
"""

import os, json, argparse, unicodedata, time
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# -------------------- Bases --------------------
# Per the current annotation schema, ONLY "Alergias medicamentosas" carries polarity.
# "Medicação Habitual" and "Diagnóstico" are flat (any Polaridade field is ignored).
TARGET_BASES_POL    = ["Alergias medicamentosas"]
TARGET_POLARITIES   = ["Positiva", "Negativa"]
TARGET_BASES_SIMPLE = ["Medicação Habitual", "Diagnóstico"]
ALL_BASES           = TARGET_BASES_POL + TARGET_BASES_SIMPLE


def _strip_accents_lower(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()


_CANON_BASE = {_strip_accents_lower(x): x for x in ALL_BASES}
_CANON_POL  = {"positiva": "Positiva", "negativa": "Negativa", "nagetiva": "Negativa"}


def canonical_base(raw: Optional[str]) -> Optional[str]:
    return _CANON_BASE.get(_strip_accents_lower(raw or ""), None)


def canonical_pol(raw: Optional[str]) -> Optional[str]:
    return _CANON_POL.get(_strip_accents_lower(raw or ""), None)


# -------------------- I/O helpers --------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_dir(model_or_output_dir: str) -> str:
    """Return the model directory to load from.
    Prefers the explicit `best/` subdirectory over the parent, so we never
    accidentally load a transient checkpoint instead of the best model.
    """
    best = os.path.join(model_or_output_dir, "best")
    if os.path.isfile(os.path.join(best, "config.json")):
        return best
    if os.path.isfile(os.path.join(model_or_output_dir, "config.json")):
        # Warn if this looks like a transient checkpoint, not the saved best model.
        if "checkpoint-" in os.path.basename(os.path.normpath(model_or_output_dir)):
            print(f"[WARN] --model_dir points at a transient checkpoint folder: "
                  f"{model_or_output_dir}\n"
                  f"       Results may not reflect the best epoch. "
                  f"Pass the `best/` subdirectory instead.")
        return model_or_output_dir
    raise FileNotFoundError(
        f"No config.json found in '{model_or_output_dir}' or '{best}'. "
        f"Check --model_dir."
    )


# -------------------- Model label maps --------------------

def normalize_label_maps(model):
    id2label_raw = getattr(model.config, "id2label", None)
    if isinstance(id2label_raw, dict):
        try:
            id2label = {int(k): v for k, v in id2label_raw.items()}
            label2id = {v: k for k, v in id2label.items()}
            return id2label, label2id
        except Exception:
            pass
    # Fallback: build from ALL_BASES (used only if config lacks id2label)
    tags = ["O"]
    for b in (TARGET_BASES_POL + TARGET_BASES_SIMPLE):
        tags += [f"B-{b}", f"I-{b}"]
    id2label = {i: t for i, t in enumerate(tags)}
    label2id = {t: i for i, t in id2label.items()}
    return id2label, label2id


# -------------------- Decoding utilities --------------------

def decode_spans_from_bio(offsets, tags: List[str]):
    """Decode BIO tag strings → list of (begin, end, raw_label_str)."""
    spans = []; cur = None; idxs = []
    for i, t in enumerate(tags):
        if t == "O" or "-" not in t:
            if cur and idxs:
                b = offsets[idxs[0]][0]; e = offsets[idxs[-1]][1]
                spans.append((b, e, cur))
            cur = None; idxs = []; continue
        pref, lab = t.split("-", 1)
        if pref == "B":
            if cur and idxs:
                b = offsets[idxs[0]][0]; e = offsets[idxs[-1]][1]
                spans.append((b, e, cur))
            cur = lab; idxs = [i]
        elif pref == "I":
            if cur == lab:
                idxs.append(i)
            else:
                cur = lab; idxs = [i]
    if cur and idxs:
        b = offsets[idxs[0]][0]; e = offsets[idxs[-1]][1]
        spans.append((b, e, cur))
    return spans


def parse_base_pol_from_raw(tag_label: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse 'Alergias medicamentosas__Positiva' or 'Medicação Habitual' etc.
    Returns (base, pol|None). Flat classes always get pol=None.
    """
    if not tag_label:
        return None, None
    if "__" in tag_label:
        left, right = tag_label.split("__", 1)
    else:
        left, right = tag_label, None
    base = canonical_base(left.strip())
    pol  = canonical_pol(right.strip()) if right is not None else None
    # Enforce: flat bases never get polarity
    if base in TARGET_BASES_SIMPLE:
        pol = None
    # Enforce: polarity bases must have a recognized polarity
    if base in TARGET_BASES_POL and pol is not None and pol not in TARGET_POLARITIES:
        pol = None
    return base, pol


def ltrim_ws(text: str, b: int, e: int) -> Tuple[int, int]:
    if b is None or e is None: return b, e
    if b < 0 or e > len(text) or b >= e: return b, e
    while b < e and text[b].isspace():
        b += 1
    return b, e


def iou_char(a, b):
    (ba, ea), (bb, eb) = a, b
    inter = max(0, min(ea, eb) - max(ba, bb))
    union = max(ea, eb) - min(ba, bb)
    return inter / union if union > 0 else 0.0


# -------------------- Metrics --------------------

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def class_key(base: Optional[str], pol: Optional[str]) -> Optional[str]:
    """Combine entity base and polarity into a single per-class key for reporting."""
    if base is None:
        return None
    if base in TARGET_BASES_POL:
        return f"{base}__{pol}" if pol else None
    return base   # Medicação Habitual / Diagnóstico


def polarity_report(pairs: List[Tuple[Optional[str], Optional[str]]],
                    classes=("Positiva", "Negativa")):
    stats = {c: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in classes}
    for g, p in pairs:
        if g in classes:
            stats[g]["support"] += 1
        for c in classes:
            if g == c and p == c:   stats[c]["tp"] += 1
            elif g != c and p == c: stats[c]["fp"] += 1
            elif g == c and p != c: stats[c]["fn"] += 1
    per_class = {}; f1s = []; sum_tp = sum_fp = sum_fn = 0
    for c in classes:
        tp, fp, fn = stats[c]["tp"], stats[c]["fp"], stats[c]["fn"]
        P, R, F = prf(tp, fp, fn)
        per_class[c] = {"precision": P, "recall": R, "f1": F, "support": stats[c]["support"]}
        f1s.append(F)
        sum_tp += tp; sum_fp += fp; sum_fn += fn
    microP, microR, microF = prf(sum_tp, sum_fp, sum_fn)
    macro = {
        "precision": float(np.mean([per_class[c]["precision"] for c in classes])) if classes else 0.0,
        "recall":    float(np.mean([per_class[c]["recall"]    for c in classes])) if classes else 0.0,
        "f1":        float(np.mean(f1s)) if f1s else 0.0,
        "support":   int(sum(stats[c]["support"] for c in classes))
    }
    micro = {"precision": microP, "recall": microR, "f1": microF, "support": macro["support"]}
    return micro, macro, per_class


def print_block(title, micro, macro, per_class):
    print(f"===== {title} =====")
    print(f"Micro  — P: {micro['precision']:.4f} | R: {micro['recall']:.4f} | "
          f"F1: {micro['f1']:.4f} | Support: {micro['support']}")
    print(f"Macro  — P: {macro['precision']:.4f} | R: {macro['recall']:.4f} | "
          f"F1: {macro['f1']:.4f} | Support: {macro['support']}")
    print("Per-class:")
    for c, m in per_class.items():
        print(f"  {c:42s}  P: {m['precision']:.4f} | R: {m['recall']:.4f} | "
              f"F1: {m['f1']:.4f} | support: {m['support']}")


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Path to model directory. Prefer passing the `best/` "
                         "subdirectory directly to avoid loading a transient checkpoint.")
    ap.add_argument("--test_json", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128,
                    help="Sliding-window stride in tokens. "
                         "Must match the stride used during training. "
                         "Set to 0 to disable (single truncated chunk, NOT recommended).")
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--clean", action="store_true",
                    help="Drop gold Alergias annotations that have no Polaridade.")
    ap.add_argument("--relaxed_iou", type=float, default=0.5)
    ap.add_argument("--score_mode", type=str, default="joint",
                    choices=["matched", "joint"])
    ap.add_argument("--pred_json", type=str, default=None)
    args = ap.parse_args()

    if args.gpu is not None and str(args.gpu).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = find_best_dir(args.model_dir)
    print(f"[INFO] Loading model from : {model_dir}")
    print(f"[INFO] Device             : {device}")
    print(f"[INFO] Test file          : {args.test_json}")
    print(f"[INFO] Stride             : {args.stride}")
    print(f"[INFO] Score mode         : {args.score_mode}")
    print(f"[INFO] Schema             : POL={TARGET_BASES_POL} | FLAT={TARGET_BASES_SIMPLE}")

    tok   = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()
    id2label, _ = normalize_label_maps(model)

    # Confirm label space matches the expected schema
    print(f"[INFO] Model label space ({len(id2label)} labels):")
    for i, lbl in sorted(id2label.items()):
        print(f"       {i:2d}  {lbl}")

    data = load_json(args.test_json)

    # Accumulators
    pairs_exact: List[Tuple[Optional[str], Optional[str]]] = []
    pairs_relax: List[Tuple[Optional[str], Optional[str]]] = []
    class_pairs_exact: List[Tuple[Optional[str], Optional[str]]] = []
    class_pairs_relax: List[Tuple[Optional[str], Optional[str]]] = []

    out_docs = []
    n_alergias_no_pol = 0

    with torch.no_grad():
        for ex in data:
            text = ex["text"]

            # Build gold list: (b, e, base, pol|None)
            gold = []
            for a in ex.get("annotations", []):
                base = canonical_base(a.get("label"))
                b, e = int(a.get("begin", -1)), int(a.get("end", -1))
                if base not in ALL_BASES or not (0 <= b < e <= len(text)):
                    continue
                if base in TARGET_BASES_POL:
                    pol = canonical_pol(a.get("Polaridade"))
                    if pol is None:
                        n_alergias_no_pol += 1
                        if args.clean:
                            continue
                else:
                    pol = None   # Medicação Habitual / Diagnóstico: polarity ignored
                gold.append((b, e, base, pol))

            # ----- Sliding-window inference -----
            full_enc    = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            full_offsets = full_enc["offset_mapping"]
            n_tokens     = len(full_offsets)
            num_labels   = model.config.num_labels
            agg_logits   = np.zeros((n_tokens, num_labels), dtype=np.float64)
            agg_count    = np.zeros(n_tokens, dtype=np.int32)

            if args.stride and args.stride > 0 and n_tokens > args.max_len - 2:
                enc = tok(
                    text,
                    return_offsets_mapping=True,
                    return_overflowing_tokens=True,
                    truncation=True,
                    max_length=args.max_len,
                    stride=args.stride,
                    padding=False,
                )
                for ch in range(len(enc["input_ids"])):
                    ipt = {k: torch.tensor(enc[k][ch]).unsqueeze(0).to(device)
                           for k in enc.keys()
                           if k not in ("offset_mapping", "overflow_to_sample_mapping")}
                    logits = model(**ipt).logits.squeeze(0).cpu().numpy()
                    ptr = 0
                    for i, (s, e) in enumerate(enc["offset_mapping"][ch]):
                        if s == 0 and e == 0:
                            continue
                        while ptr < n_tokens and full_offsets[ptr] != (s, e):
                            ptr += 1
                        if ptr >= n_tokens:
                            ptr = 0
                            while ptr < n_tokens and full_offsets[ptr] != (s, e):
                                ptr += 1
                            if ptr >= n_tokens:
                                continue
                        agg_logits[ptr] += logits[i]
                        agg_count[ptr]  += 1
                        ptr += 1
            else:
                enc = tok(text, return_offsets_mapping=True, truncation=True,
                          max_length=args.max_len, padding=False)
                ipt = {k: torch.tensor(v).unsqueeze(0).to(device)
                       for k, v in enc.items() if k != "offset_mapping"}
                logits = model(**ipt).logits.squeeze(0).cpu().numpy()
                ptr = 0
                for i, (s, e) in enumerate(enc["offset_mapping"]):
                    if s == 0 and e == 0:
                        continue
                    while ptr < n_tokens and full_offsets[ptr] != (s, e):
                        ptr += 1
                    if ptr >= n_tokens:
                        break
                    agg_logits[ptr] += logits[i]
                    agg_count[ptr]  += 1
                    ptr += 1

            covered = agg_count > 0
            agg_logits[covered] /= agg_count[covered, None]
            o_idx = next((i for i, lbl in id2label.items() if lbl == "O"), 0)
            pred_ids = agg_logits.argmax(-1).tolist()
            for i in range(n_tokens):
                if not covered[i]:
                    pred_ids[i] = o_idx

            tags = [id2label[int(pid)] for pid in pred_ids]
            raw  = decode_spans_from_bio(full_offsets, tags)

            # Decode predictions → (b, e, base, pol|None)
            preds = []
            for (b, e, raw_lab) in raw:
                base, pol = parse_base_pol_from_raw(raw_lab)
                if base not in ALL_BASES:
                    continue
                b, e = ltrim_ws(text, b, e)
                if b < e:
                    preds.append((b, e, base, pol))

            # ---- EXACT match ----
            used_gold_exact = set()
            used_pred_exact = set()
            for pi, pb in enumerate(preds):
                best = None
                for j, gb in enumerate(gold):
                    if j in used_gold_exact or pb[2] != gb[2]:
                        continue
                    if pb[0] == gb[0] and pb[1] == gb[1]:
                        best = j; break
                if best is not None:
                    used_gold_exact.add(best); used_pred_exact.add(pi)
                    if gold[best][2] in TARGET_BASES_POL:
                        pairs_exact.append((gold[best][3], pb[3]))
                    gk = class_key(gold[best][2], gold[best][3])
                    pk = class_key(pb[2], pb[3])
                    if gk is not None or pk is not None:
                        class_pairs_exact.append((gk, pk))

            # ---- RELAXED IoU@T match ----
            used_gold_relax = set()
            used_pred_relax = set()
            for pi, pb in enumerate(preds):
                best = None; best_iou = 0.0
                for j, gb in enumerate(gold):
                    if j in used_gold_relax or pb[2] != gb[2]:
                        continue
                    iou = iou_char((pb[0], pb[1]), (gb[0], gb[1]))
                    if iou >= args.relaxed_iou and iou > best_iou:
                        best, best_iou = j, iou
                if best is not None:
                    used_gold_relax.add(best); used_pred_relax.add(pi)
                    if gold[best][2] in TARGET_BASES_POL:
                        pairs_relax.append((gold[best][3], pb[3]))
                    gk = class_key(gold[best][2], gold[best][3])
                    pk = class_key(pb[2], pb[3])
                    if gk is not None or pk is not None:
                        class_pairs_relax.append((gk, pk))

            # ---- Joint scoring: FN/FP for all bases ----
            if args.score_mode == "joint":
                for j, (gb, ge, gbase, gpol) in enumerate(gold):
                    if gbase in TARGET_BASES_POL and j not in used_gold_relax:
                        pairs_relax.append((gpol, None))
                for i, (pb, pe, pbase, ppol) in enumerate(preds):
                    if pbase in TARGET_BASES_POL and i not in used_pred_relax:
                        pairs_relax.append((None, ppol))
                for j, (gb, ge, gbase, gpol) in enumerate(gold):
                    if gbase in TARGET_BASES_POL and j not in used_gold_exact:
                        pairs_exact.append((gpol, None))
                for i, (pb, pe, pbase, ppol) in enumerate(preds):
                    if pbase in TARGET_BASES_POL and i not in used_pred_exact:
                        pairs_exact.append((None, ppol))
                # Per-class joint (covers ALL bases incl. Medicação Habitual / Diagnóstico)
                for j, (gb, ge, gbase, gpol) in enumerate(gold):
                    if j in used_gold_relax: continue
                    gk = class_key(gbase, gpol)
                    if gk is not None:
                        class_pairs_relax.append((gk, None))
                for i, (pb, pe, pbase, ppol) in enumerate(preds):
                    if i in used_pred_relax: continue
                    pk = class_key(pbase, ppol)
                    if pk is not None:
                        class_pairs_relax.append((None, pk))
                for j, (gb, ge, gbase, gpol) in enumerate(gold):
                    if j in used_gold_exact: continue
                    gk = class_key(gbase, gpol)
                    if gk is not None:
                        class_pairs_exact.append((gk, None))
                for i, (pb, pe, pbase, ppol) in enumerate(preds):
                    if i in used_pred_exact: continue
                    pk = class_key(pbase, ppol)
                    if pk is not None:
                        class_pairs_exact.append((None, pk))

            # ---- Per-doc records for audit ----
            if args.pred_json:
                thr = args.relaxed_iou
                records = []
                used_gold_for_records = set()
                for (pb, pe, base, ppol) in preds:
                    best_i, best_iou = None, 0.0
                    for i, (gb, ge, gbase, gpol) in enumerate(gold):
                        if gbase != base: continue
                        iou = iou_char((pb, pe), (gb, ge))
                        if iou >= thr and iou > best_iou:
                            best_i, best_iou = i, iou
                    p_word = text[pb:pe] if 0 <= pb < pe <= len(text) else None
                    is_diag = base in TARGET_BASES_SIMPLE
                    if best_i is not None:
                        gb, ge, gbase, gpol = gold[best_i]
                        g_word = text[gb:ge] if 0 <= gb < ge <= len(text) else None
                        used_gold_for_records.add(best_i)
                        records.append({
                            "label": gbase,
                            "diagnostico": bool(is_diag),
                            "Polaridade": gpol if gbase in TARGET_BASES_POL else None,
                            "begin": int(gb), "end": int(ge), "g_word": g_word,
                            "p_label": base,
                            "p_polaridade": ppol if base in TARGET_BASES_POL else None,
                            "p_begin": int(pb), "p_end": int(pe), "p_word": p_word,
                        })
                    else:
                        records.append({
                            "label": None, "diagnostico": bool(is_diag),
                            "Polaridade": None, "begin": None, "end": None, "g_word": None,
                            "p_label": base,
                            "p_polaridade": ppol if base in TARGET_BASES_POL else None,
                            "p_begin": int(pb), "p_end": int(pe), "p_word": p_word,
                        })
                for gi, (gb, ge, gbase, gpol) in enumerate(gold):
                    if gi in used_gold_for_records: continue
                    g_word = text[gb:ge] if 0 <= gb < ge <= len(text) else None
                    records.append({
                        "label": gbase,
                        "diagnostico": bool(gbase in TARGET_BASES_SIMPLE),
                        "Polaridade": gpol if gbase in TARGET_BASES_POL else None,
                        "begin": int(gb), "end": int(ge), "g_word": g_word,
                        "p_label": None, "p_polaridade": None,
                        "p_begin": None, "p_end": None, "p_word": None,
                    })
                out_docs.append({"doc_id": ex.get("doc_id"), "predictions": records})

    # ------------- Surface any schema drift -------------
    if n_alergias_no_pol:
        print(f"\n[WARN] {n_alergias_no_pol} 'Alergias medicamentosas' gold annotation(s) "
              f"had no Polaridade and were included with pol=None. "
              f"Use --clean to drop them instead.")

    # ------------- Polarity summaries (Positiva vs Negativa) -------------
    mode_note = ("(matched spans only)" if args.score_mode == "matched"
                 else "(joint extraction + polarity)")
    print(f"\n=== Polarity evaluation {mode_note} ===")
    mic_e, mac_e, cls_e = polarity_report(pairs_exact)
    print_block("EXACT (original indices)", mic_e, mac_e, cls_e)
    mic_r, mac_r, cls_r = polarity_report(pairs_relax)
    print_block(f"RELAXED IoU@{args.relaxed_iou:.2f}", mic_r, mac_r, cls_r)

    # ------------- Per BIO+polarity class (4 numbers) -------------
    seen_classes = sorted({k for pair in (class_pairs_exact + class_pairs_relax)
                           for k in pair if k is not None})
    if seen_classes:
        print(f"\n=== Per-class evaluation {mode_note} ===")
        mic_ce, mac_ce, cls_ce = polarity_report(class_pairs_exact, classes=tuple(seen_classes))
        print_block("EXACT (per BIO+polarity class)", mic_ce, mac_ce, cls_ce)
        mic_cr, mac_cr, cls_cr = polarity_report(class_pairs_relax, classes=tuple(seen_classes))
        print_block(f"RELAXED IoU@{args.relaxed_iou:.2f} (per BIO+polarity class)", mic_cr, mac_cr, cls_cr)
    else:
        mic_ce = mac_ce = cls_ce = mic_cr = mac_cr = cls_cr = None

    # ------------- Optional JSON dump -------------
    if args.pred_json:
        payload = {
            "timestamp":              time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_path":             model_dir,
            "test_file":              args.test_json,
            "relaxed_iou":            float(args.relaxed_iou),
            "score_mode":             args.score_mode,
            "bases_with_polarity":    TARGET_BASES_POL,
            "bases_without_polarity": TARGET_BASES_SIMPLE,
            "polarity_exact_micro":   mic_e,
            "polarity_exact_macro":   mac_e,
            "polarity_relaxed_micro": mic_r,
            "polarity_relaxed_macro": mac_r,
            "per_class_exact_micro":  mic_ce,
            "per_class_exact_macro":  mac_ce,
            "per_class_exact":        cls_ce,
            "per_class_relaxed_micro": mic_cr,
            "per_class_relaxed_macro": mac_cr,
            "per_class_relaxed":       cls_cr,
            "docs":                   out_docs,
        }
        with open(args.pred_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.pred_json}")


if __name__ == "__main__":
    main()