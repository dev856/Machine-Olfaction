# SmellNet Data Dictionary

## 1) Dataset Overview

- Dataset root path:
- Date inspected:
- Inspector:
- Data subsets found (example: base_data, mixture_data, gcms_data, gcms_processed, text_data):
- Notes:

## 2) File Inventory

| Subset/Folder | File path (relative) | File meaning/purpose | Row count | Column count | One row = what? (reading/sample/trial/other) | Notes |
|---|---|---|---:|---:|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## 3) Per-File Schema Details

### File: TODO

- Relative path:
- Purpose of this file:
- Shape (rows, columns):
- Primary key candidate(s):
- Trial/sample identifier column(s):
- Timestamp column:
- Label column(s):
- Class name column(s):
- Sensor column(s):
- Non-sensor metadata column(s):
- Target task suggested by this file (classification/regression/other):
- Notes:

#### Column Dictionary

| Column name | Data type (observed) | Role (sensor/timestamp/label/id/metadata/target) | Example value | Missing count | Missing % | Unit (if available) | Notes |
|---|---|---|---|---:|---:|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## 4) Sensor Columns Summary (Across Files)

| File path | Sensor columns (exact names) | Sensor count | Any naming pattern? | Unit/source info | Notes |
|---|---|---:|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

## 5) Timestamp Summary (Across Files)

| File path | Timestamp column name | Raw format example | Parsed format (if tested) | Timezone info | Monotonic within trial? | Sampling interval known? | Notes |
|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## 6) Label & Class Summary

### 6.1 Base substances (if present)

| Label column | Class column | Unique classes | Example class values | Encoding type (string/int) | Notes |
|---|---|---:|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

### 6.2 Mixtures (if present)

| Mixture label column(s) | Base component columns | Ratio/proportion columns | Encoding details | How mixtures differ from base labels | Notes |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

## 7) Missing Values Summary

| File path | Columns with missing values | Total missing cells | Missingness pattern (random/by-column/by-trial) | Action candidate (drop/impute/keep) | Notes |
|---|---|---:|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

## 8) Units and Measurement Notes

| Column(s) | Unit found | Where unit was found (README/header/paper/none) | Confidence (high/medium/low) | Notes |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO |

## 9) Open Questions

- TODO
- TODO
- TODO

## 10) Decisions for Next Step (Preprocessing Planning)

- Keep/drop columns:
- Label definition for baseline:
- File granularity assumption (one file = one sample/trial/many readings):
- Timestamp handling plan:
- Missing value handling plan:
- TODO items before coding loader: