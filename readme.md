# Portuguese Emergency Room Clinical NER Dataset and Baselines

This repository contains the dataset, prompts, training scripts, baseline experiments, and evaluation outputs used in the paper **"NER Models for Portuguese Emergency Room Notes: Extracting Diagnoses, Medication Allergies, and Usual Medications"**.

The project supports research on Portuguese clinical Named Entity Recognition (NER) in Emergency Room (ER) admission notes. It focuses on three clinically relevant information types used in ER handovers and decision support:

- **Principal diagnosis** (`Diagnóstico`)
- **Usual medication** (`Medicação Habitual`)
- **Medication allergies** (`Alergias medicamentosas`), including positive and negative polarity

The dataset is composed of synthetic and physician-written Portuguese ER admission notes, annotated with entity spans and enriched with terminology codes for interoperability.

> **Important:** The clinical notes in this repository are fictional or synthetic. They are intended for research and benchmarking only, and the models/scripts are not intended for direct clinical use without appropriate validation, governance, and regulatory review.

---

## Repository structure

```text
ER_NER/
├── dataset/
│   ├── train.json
│   ├── val.json
│   └── test-real.json
├── encoders_training_testing/
│   ├── train.py
│   ├── test-per-class.py
│   ├── train-run.sh
│   ├── test_sub.sh
│   └── additional_details.md
├── generative_testing/
│   ├── ER_NER_baseline.ipynb
│   └── er_gen_amalia.py
├── prompts/
│   ├── prompt_synthetic_data_gen.md
│   └── prompt_generative_NER_extraction.md
├── results/
│   ├── polarity_eval_alldis-test-real.json
│   └── polarity_eval_alldis-test-real-biobert-all.json
├── images/
│   ├── annotation_scheme.png
│   ├── dataset_split_count.png
│   ├── annotation_example_diagnosis.png
│   └── usualmedication+medicationallergies_example.png
└── README.md
```

---

## Dataset overview

The dataset contains **300 Portuguese ER admission notes**, divided into:

| Split | Documents | Purpose |
|---|---:|---|
| `train.json` | 257 | Encoder model fine-tuning |
| `val.json` | 28 | Validation and early stopping |
| `test-real.json` | 15 | Evaluation on physician-written/validated notes |

The annotation schema follows the paper's ER-focused information extraction task:

| Entity label in JSON | Meaning | Additional fields |
|---|---|---|
| `Diagnóstico` | Principal diagnosis for the current admission episode | `ICD10` |
| `Medicação Habitual` | Patient's chronic/usual medication | `EDQM` field contains the SNOMED CT code in this release |
| `Alergias medicamentosas` | Medication allergy or explicit absence of medication allergies | `Polaridade`, `ATC` |

For medication allergies, `Polaridade` is one of:

- `Positiva`: the patient has a medication allergy
- `Negativa`: the note explicitly states absence of medication allergies

### JSON format

Each dataset file is a list of documents. Each document contains the raw note text and character-level annotations:

```json
{
  "doc_id": 1,
  "text": "...",
  "annotations": [
    {
      "label": "Medicação Habitual",
      "diagnostico": null,
      "Polaridade": null,
      "ICD10": null,
      "EDQM": "376701008",
      "begin": 924,
      "end": 942
    }
  ]
}
```

The `begin` and `end` fields are character offsets into `text`, with `end` as an exclusive offset. The span text can be recovered with:

```python
span = document["text"][annotation["begin"]:annotation["end"]]
```

---

## Task formulation

The encoder models formulate the problem as BIO sequence labeling over four effective classes:

1. `Alergias medicamentosas__Positiva`
2. `Alergias medicamentosas__Negativa`
3. `Medicação Habitual`
4. `Diagnóstico`

The resulting BIO label space contains one `O` label plus `B-` and `I-` labels for each class.

Long notes are handled with sliding-window tokenization:

- Maximum sequence length: `512`
- Stride: `128`
- Overlapping-window logits are averaged during inference before decoding

Class imbalance is handled using:

- Inverse-frequency class weights in the cross-entropy loss
- Document-level oversampling of notes containing `Alergias medicamentosas__Negativa`

More implementation details are available in [`encoders_training_testing/additional_details.md`](encoders_training_testing/additional_details.md).

---


## Ethical and clinical-use statement

This repository is released for research on clinical NLP, synthetic data generation, and Portuguese medical information extraction. Although the dataset is synthetic/fictional, clinical NLP systems can still cause harm if deployed without adequate validation. Do not use the models or code in clinical workflows without institutional approval, data-governance review, and prospective clinical validation.
