# Portuguese Emergency Room Clinical NER

**Dataset, models, and baselines for extracting diagnoses, medication allergies, and usual medications from Portuguese ER admission notes.**

This repository accompanies the paper *"NER Models for Portuguese Emergency Room Notes: Extracting Diagnoses, Medication Allergies, and Usual Medications"* and contains everything needed to reproduce the experiments: the annotated dataset, fine-tuning and evaluation scripts for encoder models, generative LLM baselines, prompts, and pre-computed results.

> **Clinical use disclaimer:** The clinical notes in this dataset are fictional or synthetic and are intended for research and benchmarking only. The models and scripts are **not** validated for direct clinical use. Deployment in any clinical workflow requires institutional approval, data-governance review, and prospective clinical validation.

---

## Overview

Emergency Room (ER) handovers require rapid identification of a patient's principal diagnosis, usual medications, and medication allergies. This project develops and evaluates specialised NER models for these three entities in Portuguese, a language underrepresented in clinical NLP resources.

**Key contributions:**

- A **synthetic dataset** of 300 Portuguese ER admission notes (275 LLM-generated with Llama 3.3 + 15 physician-validated), covering eight medical specialties.
- **Two-layer annotation**: entity spans for the three target classes, plus mappings to standard terminologies (ICD-10 for diagnoses, ATC for allergies, SNOMED CT for usual medications).
- **Fine-tuned NER models**: BioBERT-PT and MediAlbertina, benchmarked against few-shot Gemini and Gemma baselines.

### Results summary (macro F1 on physician-validated test set)

| Model | Exact match | IoU ≥ 0.50 |
|---|:---:|:---:|
| BioBERT-PT | **0.75** | **0.82** |
| MediAlbertina | 0.70 | 0.80 |
| Gemma 4 (open-weight) | 0.48 | 0.63 |
| Gemini 2.5 Flash Lite (closed-weight) | 0.35 | 0.44 |

Encoder models substantially outperform generative baselines. Principal diagnosis is the most challenging class; usual medication extraction achieves the strongest performance across all models.

---

## Repository structure

```
ER_NER/
├── dataset/
│   ├── train.json                          # 257 documents for fine-tuning
│   ├── val.json                            # 28 documents for validation
│   ├── test-real.json                      # 15 physician-validated documents for evaluation
│   └── ER_NER_Dataset_Characterization.ipynb  # Dataset statistics and analysis
│
├── encoders_training_testing/
│   ├── train.py                            # Fine-tuning script (HuggingFace Transformers)
│   ├── train-run.sh                        # Training launcher with hyperparameter config
│   ├── test-per-class.py                   # Evaluation script (exact match + IoU@0.5)
│   ├── test_sub.sh                         # Evaluation launcher
│   └── additional_details.md              # Full training and evaluation details
│
├── generative_testing/
│   ├── ER_NER_baseline__cleaned.ipynb      # Gemini / Gemma few-shot NER extraction
│   └── ER_NER_evaluation.ipynb            # Evaluation of generative model outputs
│
├── prompts/
│   ├── prompt_synthetic_data_gen.md        # Prompt template used to generate clinical notes
│   └── prompt_generative_NER_extraction.md # Prompt template used for generative NER baselines
│
├── results/
│   ├── biobertpt.json                      # BioBERT-PT predictions on test set
│   ├── medialbertina.json                  # MediaAlbertina predictions on test set
│   ├── gemini-2.5-flash-lite/              # Per-document Gemini predictions (JSON + HTML) on test set
|   ├──  gemma-4-31b-it/		     # Per-document Gemma predictions (JSON + HTML) on test set

│
├── images/
│   ├── annotation_scheme.png
│   ├── dataset_split_count.png
│   ├── annotation_example_diagnosis.png
│   └── usualmedication+medicationallergies_example.png
│
└── README.md
```

---

## Dataset

### Splits

| Split | File | Documents | Annotated spans | Notes |
|---|---|:---:|:---:|---|
| Train | `train.json` | 257 | 1,492 | LLM-generated; used for fine-tuning |
| Validation | `val.json` | 28 | 166 | LLM-generated; used for early stopping |
| Test | `test-real.json` | 15 | 86 | Physician-validated; held out for evaluation |

The train/validation split uses iterative multi-label stratification to ensure proportional class representation. The test set is composed exclusively of the 15 physician-validated notes, providing a close-to-real-world evaluation benchmark.

### Annotation and terminology mappings

![Annotation Scheme](images/annotation_scheme.png)

| JSON label | Entity | Terminology | Notes |
|---|---|---|---|
| `Diagnóstico` | Principal diagnosis for the current ER episode | ICD-10 | One per note; specific codes only |
| `Medicação Habitual` | Patient's chronic/usual medication | SNOMED CT | Includes dose and administration instructions when present |
| `Alergias medicamentosas` | Medication allergy, or explicit absence of allergy | ATC | Has a `Polaridade` field: `Positiva` or `Negativa` |



### Annotation notes

- **Medication allergies:** the annotated markable is the allergenic agent itself (e.g. `"penicillin"` in *"reports a known allergy to penicillin"*). For negated contexts (e.g. *"has no drug allergies"*), the markable is the negated phrase (e.g. `"drug allergies"`) with `Polaridade = Negativa`.
- **Principal diagnosis:** the most specific ICD-10 code is assigned where possible. Overly generic descriptions (e.g. *"cardiovascular disease"*) are not coded.
- **Usual medication:** spans include the medication name plus dosage and administration instructions when present. SNOMED CT codes are stored in the `EDQM` field in this release.

---

## Models

The task is formulated as **BIO sequence labelling** over four effective classes:

| BIO class | Description |
|---|---|
| `Alergias medicamentosas__Positiva` | Positive medication allergy |
| `Alergias medicamentosas__Negativa` | Explicit absence of medication allergy |
| `Medicação Habitual` | Usual/chronic medication |
| `Diagnóstico` | Principal diagnosis |

This yields **9 output labels**: `O` plus `B-` and `I-` for each of the four classes.

Two base models are supported:

| Model | HuggingFace identifier |
|---|---|
| MediAlbertina | `portugueseNLP/medialbertina_pt-pt_900m` |
| BioBERT-PT | `pucpr/biobertpt-all` |

### Training

Configure paths and hyperparameters in `encoders_training_testing/train-run.sh`, then run:

```bash
cd encoders_training_testing
bash train-run.sh
```

### Evaluation

```bash
cd encoders_training_testing
python test-per-class.py \
  --model_dir {OUTPUT_DIR}/best \
  --test_json ../dataset/test-real.json \
  --max_len 512 \
  --stride 128 \
  --score_mode joint \
  --pred_json predictions.json
```

Two span matching criteria are reported:

- **Exact match:** predicted and gold spans must have identical character-level start and end indices.
- **Relaxed match (IoU ≥ 0.50):** the character-level overlap between predicted and gold spans must meet a minimum intersection-over-union threshold of 50%.

Evaluation runs in **joint extraction + polarity** mode (`--score_mode joint`): missed and spurious spans are penalised in addition to polarity errors.

---

## Generative baselines

Few-shot generative NER experiments are in `generative_testing/ER_NER_baseline__cleaned.ipynb`. Both baselines use [LangExtract](https://github.com/agoel00/langextract) to obtain structured outputs.

For each test document, the most semantically similar training document (by cosine similarity of text embeddings) is retrieved and used as a dynamic few-shot example.

| Model | Type | HuggingFace / API identifier |
|---|---|---|
| Gemini 2.5 Flash Lite | Closed-weight (API) | `gemini-2.5-flash-lite` |
| Gemma 4 | Open-weight (local) | `gemma-4-31B-it` |

> **Privacy note:** closed-weight API models are unsuitable for real ER settings due to data governance constraints. Open-weight models can be deployed locally within hospital infrastructure.

Pre-computed predictions for Gemini are available in `results/gemini-2.5-flash-lite/`.

---

## Prompts

| File | Purpose |
|---|---|
| `prompts/prompt_synthetic_data_gen.md` | Template used with Llama 3.3 to generate synthetic clinical notes. Parameterised by `{medical specialty}` and `{allergy}` (presence/absence), with a physician-validated note as `{example}`. |
| `prompts/prompt_generative_NER_extraction.md` | Extraction prompt for the generative baselines, defining the three entity classes and extraction rules. |

---

## Dataset construction

1. Five physicians from four specialties each wrote one fictional ER admission note.
2. Fifteen variations were generated from these examples using the synthetic data generation prompt, then reviewed and validated by the same physicians.
3. 275 additional notes were generated using Llama 3.3, with the 15 validated notes as few-shot examples. Medical specialty and allergy presence/absence were varied systematically across generations.
4. The resulting 275 synthetic notes were combined with the 15 physician-validated notes for annotation.

**Quality evaluation:** two independent physicians assessed 60 synthetic notes each using a six-question Likert protocol. The notes scored positively on medication clarity (Q2) and allergy identification (Q5), with moderate scores on diagnosis specificity (Q4). See the paper for the full evaluation results and inter-annotator agreement (Krippendorff's α).

**Annotation** was performed by a PhD student in Linguistics with a pharmaceutical background, using a layered approach (markables → allergy codes → diagnosis codes → medication codes).

---

## Citation

If you use this dataset, models, or code in your work, please cite:

```bibtex
@inproceedings{ernermodels2026,
  title     = {NER Models for Portuguese Emergency Room Notes: Extracting Diagnoses, Medication Allergies, and Usual Medications},
  author    = {Anonymous},
  booktitle = {Anonymous Submission},
  year      = {2026}
}
```

*This entry will be updated with the full citation upon publication.*

---

