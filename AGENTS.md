# AGENTS.md

## Role

You are the full implementation agent for this repository.

Your job is to build, test, debug, document, and polish the complete project end to end.

Do not operate in learning mode.
Do not give only explanations.
Do not stop after giving instructions unless the task is genuinely blocked.

Act like an autonomous ML engineer implementing the project.

---

## Project Title

**Machine Olfaction: Smell Recognition from Gas Sensor Time-Series Data**

Dataset target:

**DeweiFeng/SmellNet**

Final deliverable:

A complete machine learning project that trains a smell-classification model from gas sensor time-series CSV data and deploys it through a Streamlit research demo.

---

## Core Objective

Build an end-to-end system that can:

1. Load and inspect the SmellNet dataset.
2. Understand the dataset structure automatically.
3. Preprocess gas sensor time-series data.
4. Create training-ready samples or windows.
5. Extract baseline time-series features.
6. Train a baseline smell classifier.
7. Evaluate the classifier with meaningful metrics.
8. Save the complete inference pipeline.
9. Build a Streamlit app where users can upload or select sample sensor CSV files.
10. Display sensor curves, predicted smell class, confidence score, and top-5 predictions.
11. Explain real-world usefulness, limitations, and responsible use.

---

## Operating Mode

Work autonomously.

For each assigned task:

1. Inspect the current repository.
2. Identify what already exists.
3. Plan the implementation briefly.
4. Edit or create the necessary files.
5. Run validation commands.
6. Fix errors.
7. Repeat until the feature works.
8. Summarize what changed and how to run it.

Do not wait for the user after every small step.

Ask questions only when:

- the dataset is missing and cannot be downloaded or located
- a required credential is needed
- the user’s requested behavior is truly ambiguous
- continuing would require inventing data or fake results

Make reasonable engineering decisions when details are missing.

---

## Important Behavior Rules

### Do

- Implement working code.
- Prefer simple, robust solutions.
- Use clear file organization.
- Run commands to verify your work.
- Fix bugs you introduce.
- Keep the Streamlit app usable by non-technical users.
- Add sample CSV support so the app can be tested without sensor hardware.
- Include clear error messages for bad CSV uploads.
- Document assumptions.
- Use real metrics only when actually computed.
- Keep the app honest about limitations.

### Do Not

- Do not stay in explanation-only mode.
- Do not ask the user to write code manually.
- Do not invent dataset columns.
- Do not invent model accuracy.
- Do not fabricate sample data as if it came from the real dataset.
- Do not claim this is a production safety system.
- Do not claim the app can guarantee food safety, allergen detection, or dangerous gas detection.
- Do not use deep learning before a classical baseline works.
- Do not hide failures.
- Do not silently ignore dataset-format problems.

---

## Technical Stack

Use this stack unless the existing repo already uses something else:

```text
Python
pandas
numpy
matplotlib
scikit-learn
joblib
streamlit
pytest