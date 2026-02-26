# Atsbuild

Python-based ATS evaluation and resume optimization engine.

## What it does
- Parses resume input from `.txt/.md` and `.pdf`.
- Extracts structured requirements from a Job Description.
- Computes ATS compatibility score.
- Identifies missing required and preferred skills.
- Rewrites experience bullets for stronger JD alignment.
- Generates:
  - `ats_scoring_report.json`
  - `optimized_resume.json`
  - `ats_friendly_resume.tex`

## Usage
```bash
python3 ats_engine.py \
  --resume-text sample_resume.txt \
  --job-description sample_jd.txt \
  --output-dir outputs
```

Or with a PDF resume:
```bash
python3 ats_engine.py \
  --resume-pdf resume.pdf \
  --job-description jd.txt \
  --output-dir outputs
```

> PDF input requires `pypdf` (`pip install pypdf`).

## Notes
- The skill catalog includes multiple technologies and programming languages (including **Python**) and is used for JD/resume matching.
- Output JSON is structured for downstream integrations.
