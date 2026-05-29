Extract structured clinical entities from emergency room admission notes.

Extract only information explicitly present in the text. Do not infer, normalize, or add entities that are not stated.

## Entity classes to extract

### 1. `principal_diagnosis`

- The main clinically specific diagnosis associated with the current admission or clinical episode.
- Extract the full diagnosis span as written.
- If several principal diagnoses appear listed, extract only the first.
- Do not extract vague background conditions, symptoms, procedures, or generic disease categories unless clearly presented as the principal diagnosis.

### 2. `usual_medication`

- The patient’s chronic, habitual, home, or usual medication.
- Extract each medication separately.
- The span should include the medication name and, when present, dose, formulation, route, frequency, timing, and administration instructions.
- Do not extract medications prescribed only for the current emergency episode unless they are explicitly described as usual/chronic medication.

### 3. `drug_allergy`

- Medication allergies and explicit absence of medication allergies.
- For positive allergy statements, extract the allergenic medication or drug class.
- For negative allergy statements, extract the span indicating absence of drug allergies.
- Do not include the negation marker in the span (e.g. `sem`).
- Assign polarity:
  - `positive`: a specific drug allergy is present.
  - `negative`: the note explicitly states no known drug allergies.

## General extraction rules

- Preserve original wording, spelling, abbreviations, and accents.
- Extract minimal but complete spans.
- Do not merge separate medications or allergies into one entity.
- Return no entity when the information is absent.
- Do not extract non-drug allergies.
- Do not extract past medical history unless it is clearly the principal diagnosis for the current admission.