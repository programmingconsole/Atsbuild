#!/usr/bin/env python3
"""ATS evaluation and resume optimization engine.

Features:
1) Parse resume input from text, markdown, or PDF.
2) Extract structured job requirements from a Job Description.
3) Compare resume against requirements.
4) Calculate ATS compatibility score.
5) Identify missing skills.
6) Rewrite experience bullets for stronger alignment.
7) Emit ATS scoring report JSON, optimized resume JSON, and ATS-friendly LaTeX.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_SKILL_CATALOG: Dict[str, List[str]] = {
    "programming_languages": [
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "go",
        "ruby",
        "php",
        "sql",
        "scala",
        "r",
        "rust",
        "kotlin",
        "swift",
    ],
    "frameworks": [
        "django",
        "flask",
        "fastapi",
        "react",
        "angular",
        "vue",
        "spring",
        "node.js",
        "nodejs",
        "express",
        "pytorch",
        "tensorflow",
        "pandas",
        "numpy",
    ],
    "cloud_devops": [
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "terraform",
        "jenkins",
        "github actions",
        "ci/cd",
        "linux",
    ],
    "data_ml": [
        "machine learning",
        "deep learning",
        "nlp",
        "data analysis",
        "data engineering",
        "etl",
        "spark",
        "hadoop",
        "tableau",
        "power bi",
    ],
    "soft_skills": [
        "communication",
        "leadership",
        "collaboration",
        "stakeholder management",
        "problem solving",
        "mentoring",
        "agile",
        "scrum",
    ],
}


ACTION_VERBS = [
    "Engineered",
    "Implemented",
    "Optimized",
    "Automated",
    "Led",
    "Delivered",
    "Built",
    "Designed",
]


@dataclass
class ParsedResume:
    raw_text: str
    bullets: List[str]
    sections: Dict[str, str]


@dataclass
class JobRequirements:
    required_skills: List[str]
    preferred_skills: List[str]
    responsibilities: List[str]
    qualifications: List[str]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "PDF parsing requires 'pypdf'. Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def parse_resume_input(text: str) -> ParsedResume:
    lines = [line.strip() for line in text.splitlines()]
    bullets = [
        line.lstrip("-•* ").strip()
        for line in lines
        if re.match(r"^\s*[-•*]\s+", line) and len(line.strip()) > 2
    ]

    sections: Dict[str, str] = {}
    current_header = "general"
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if content:
            sections[current_header] = sections.get(current_header, "") + (
                "\n" + content if current_header in sections else content
            )
        buffer = []

    for line in lines:
        if re.match(r"^[A-Z][A-Za-z &/]{2,30}$", line):
            flush()
            current_header = line.lower().replace(" ", "_")
            continue
        buffer.append(line)
    flush()

    return ParsedResume(raw_text=text, bullets=bullets, sections=sections)


def extract_job_requirements(job_description: str) -> JobRequirements:
    jd_lower = job_description.lower()

    required_signals = ["required", "must", "minimum", "mandatory"]
    preferred_signals = ["preferred", "nice to have", "plus", "bonus"]

    required_skills: List[str] = []
    preferred_skills: List[str] = []

    for _, skills in DEFAULT_SKILL_CATALOG.items():
        for skill in skills:
            if re.search(rf"\b{re.escape(skill)}\b", jd_lower):
                window_matches = re.finditer(
                    rf"(.{{0,80}}\b{re.escape(skill)}\b.{{0,80}})", jd_lower
                )
                matched_required = False
                for wm in window_matches:
                    window = wm.group(1)
                    if any(s in window for s in required_signals):
                        required_skills.append(skill)
                        matched_required = True
                        break
                if not matched_required:
                    preferred_skills.append(skill)

    required_skills = sorted(set(required_skills))
    preferred_skills = sorted(set(preferred_skills) - set(required_skills))

    responsibilities = extract_bullets_by_heading(
        job_description,
        ["responsibilities", "what you'll do", "you will", "role overview"],
    )
    qualifications = extract_bullets_by_heading(
        job_description,
        ["qualifications", "requirements", "what you bring", "must have"],
    )

    if not qualifications:
        qualifications = find_requirement_like_lines(job_description)

    return JobRequirements(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        responsibilities=responsibilities,
        qualifications=qualifications,
    )


def extract_bullets_by_heading(text: str, heading_keywords: Sequence[str]) -> List[str]:
    lines = text.splitlines()
    capture = False
    captured: List[str] = []

    for line in lines:
        normalized = line.strip().lower()
        if any(keyword in normalized for keyword in heading_keywords):
            capture = True
            continue
        if capture and re.match(r"^[A-Z][A-Za-z ]{2,30}:?$", line.strip()):
            break
        if capture and re.match(r"^\s*[-•*]\s+", line):
            captured.append(line.lstrip("-•* ").strip())

    return captured


def find_requirement_like_lines(text: str) -> List[str]:
    req_lines = []
    for line in text.splitlines():
        low = line.lower().strip()
        if any(token in low for token in ["years", "experience", "required", "must"]):
            req_lines.append(line.strip("-•* "))
    return [line for line in req_lines if line]


def normalize_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9+#.]+", text.lower())


def evaluate_ats(parsed_resume: ParsedResume, requirements: JobRequirements) -> Dict:
    resume_text_lower = parsed_resume.raw_text.lower()

    req_hits = [s for s in requirements.required_skills if re.search(rf"\b{re.escape(s)}\b", resume_text_lower)]
    pref_hits = [s for s in requirements.preferred_skills if re.search(rf"\b{re.escape(s)}\b", resume_text_lower)]

    required_coverage = (len(req_hits) / len(requirements.required_skills) * 100) if requirements.required_skills else 100
    preferred_coverage = (len(pref_hits) / len(requirements.preferred_skills) * 100) if requirements.preferred_skills else 100

    jd_tokens = set(normalize_tokens(" ".join(requirements.qualifications + requirements.responsibilities)))
    resume_tokens = set(normalize_tokens(parsed_resume.raw_text))
    token_overlap = len(jd_tokens & resume_tokens) / len(jd_tokens) * 100 if jd_tokens else 100

    ats_score = round(0.55 * required_coverage + 0.2 * preferred_coverage + 0.25 * token_overlap, 2)
    missing_required = sorted(set(requirements.required_skills) - set(req_hits))
    missing_preferred = sorted(set(requirements.preferred_skills) - set(pref_hits))

    return {
        "ats_score": ats_score,
        "breakdown": {
            "required_skills_coverage": round(required_coverage, 2),
            "preferred_skills_coverage": round(preferred_coverage, 2),
            "jd_resume_keyword_overlap": round(token_overlap, 2),
        },
        "matches": {
            "required_skills": sorted(req_hits),
            "preferred_skills": sorted(pref_hits),
        },
        "missing": {
            "required_skills": missing_required,
            "preferred_skills": missing_preferred,
        },
        "recommendations": generate_recommendations(missing_required, missing_preferred),
    }


def generate_recommendations(missing_required: Sequence[str], missing_preferred: Sequence[str]) -> List[str]:
    recs = []
    if missing_required:
        recs.append(
            f"Add evidence of required skills: {', '.join(missing_required)} (projects, quantified outcomes, certifications)."
        )
    if missing_preferred:
        recs.append(
            f"Add optional strengths where applicable: {', '.join(missing_preferred)}."
        )
    if not recs:
        recs.append("Resume aligns well; focus on quantifying impact and tailoring role-specific outcomes.")
    recs.append("Use ATS-friendly section headers: Summary, Skills, Experience, Education, Certifications.")
    return recs


def rewrite_bullets(parsed_resume: ParsedResume, requirements: JobRequirements, limit: int = 8) -> List[str]:
    skills_to_inject = requirements.required_skills[:6] or requirements.preferred_skills[:4]
    if not parsed_resume.bullets:
        return [
            f"{ACTION_VERBS[i % len(ACTION_VERBS)]} solutions using {skills_to_inject[i % len(skills_to_inject)] if skills_to_inject else 'core technologies'}, improving team delivery outcomes by measurable KPIs."
            for i in range(min(limit, 4))
        ]

    rewritten = []
    for idx, bullet in enumerate(parsed_resume.bullets[:limit]):
        verb = ACTION_VERBS[idx % len(ACTION_VERBS)]
        clean = re.sub(r"^(Built|Implemented|Managed|Led|Developed|Created|Designed|Optimized|Automated)\s+", "", bullet, flags=re.IGNORECASE).strip()
        skill = skills_to_inject[idx % len(skills_to_inject)] if skills_to_inject else None

        if skill and skill.lower() not in clean.lower():
            new_bullet = f"{verb} {clean} using {skill}, improving reliability, scalability, and business impact."
        else:
            new_bullet = f"{verb} {clean}; delivered measurable improvements in quality and delivery velocity."
        rewritten.append(new_bullet)

    return rewritten


def build_optimized_resume_json(parsed_resume: ParsedResume, requirements: JobRequirements, ats_report: Dict) -> Dict:
    optimized_bullets = rewrite_bullets(parsed_resume, requirements)

    return {
        "header": {
            "name": "Candidate Name",
            "email": "candidate@email.com",
            "phone": "+1-000-000-0000",
            "location": "City, Country",
            "links": ["LinkedIn", "GitHub", "Portfolio"],
        },
        "summary": "Results-driven professional optimized for role alignment with quantified achievements and ATS keyword coverage.",
        "skills": {
            "required_matched": ats_report["matches"]["required_skills"],
            "required_missing": ats_report["missing"]["required_skills"],
            "preferred_matched": ats_report["matches"]["preferred_skills"],
            "preferred_missing": ats_report["missing"]["preferred_skills"],
        },
        "experience": [
            {
                "title": "Relevant Role Title",
                "company": "Company Name",
                "duration": "YYYY - YYYY",
                "bullets": optimized_bullets,
            }
        ],
        "education": [{"degree": "Degree", "institution": "Institution", "year": "YYYY"}],
        "certifications": [],
    }


def escape_latex(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def build_latex_resume(optimized_resume: Dict) -> str:
    header = optimized_resume["header"]
    skills = optimized_resume["skills"]
    experience = optimized_resume["experience"][0]

    bullets_latex = "\n".join(
        [f"\\item {escape_latex(b)}" for b in experience["bullets"]]
    )

    return f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=0.8in]{{geometry}}
\\usepackage[hidelinks]{{hyperref}}
\\begin{{document}}
\\begin{{center}}
    {{\\LARGE \\textbf{{{escape_latex(header['name'])}}}}}\\\\
    {escape_latex(header['location'])} $|$ {escape_latex(header['phone'])} $|$ \\href{{mailto:{header['email']}}}{{{escape_latex(header['email'])}}}
\\end{{center}}

\\section*{{Summary}}
{escape_latex(optimized_resume['summary'])}

\\section*{{Skills}}
\\textbf{{Matched Required:}} {escape_latex(', '.join(skills['required_matched']) or 'N/A')}\\\\
\\textbf{{Missing Required:}} {escape_latex(', '.join(skills['required_missing']) or 'N/A')}\\\\
\\textbf{{Matched Preferred:}} {escape_latex(', '.join(skills['preferred_matched']) or 'N/A')}

\\section*{{Experience}}
\\textbf{{{escape_latex(experience['title'])}}} - {escape_latex(experience['company'])} ({escape_latex(experience['duration'])})
\\begin{{itemize}}
{bullets_latex}
\\end{{itemize}}

\\section*{{Education}}
\\textbf{{{escape_latex(optimized_resume['education'][0]['degree'])}}}, {escape_latex(optimized_resume['education'][0]['institution'])} ({escape_latex(optimized_resume['education'][0]['year'])})

\\end{{document}}
"""


def resolve_input(resume_text_path: str | None, resume_pdf_path: str | None) -> str:
    if resume_text_path:
        return read_text_file(Path(resume_text_path))
    if resume_pdf_path:
        return read_pdf_file(Path(resume_pdf_path))
    raise ValueError("Provide either --resume-text or --resume-pdf")


def run_engine(resume_text: str, job_description: str) -> Tuple[Dict, Dict, str]:
    parsed_resume = parse_resume_input(resume_text)
    requirements = extract_job_requirements(job_description)
    ats_report = evaluate_ats(parsed_resume, requirements)
    optimized_resume = build_optimized_resume_json(parsed_resume, requirements, ats_report)
    latex_resume = build_latex_resume(optimized_resume)
    return ats_report, optimized_resume, latex_resume


def main() -> None:
    parser = argparse.ArgumentParser(description="ATS evaluation and resume optimization engine")
    parser.add_argument("--resume-text", type=str, help="Path to resume text/markdown file")
    parser.add_argument("--resume-pdf", type=str, help="Path to resume PDF file")
    parser.add_argument("--job-description", type=str, required=True, help="Path to job description text file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for generated files")
    args = parser.parse_args()

    resume_text = resolve_input(args.resume_text, args.resume_pdf)
    jd_text = read_text_file(Path(args.job_description))

    ats_report, optimized_resume, latex_resume = run_engine(resume_text, jd_text)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "ats_scoring_report.json").write_text(json.dumps(ats_report, indent=2), encoding="utf-8")
    (output_dir / "optimized_resume.json").write_text(json.dumps(optimized_resume, indent=2), encoding="utf-8")
    (output_dir / "ats_friendly_resume.tex").write_text(latex_resume, encoding="utf-8")

    print(json.dumps({
        "ats_scoring_report": str(output_dir / "ats_scoring_report.json"),
        "optimized_resume_json": str(output_dir / "optimized_resume.json"),
        "latex_resume": str(output_dir / "ats_friendly_resume.tex"),
    }, indent=2))


if __name__ == "__main__":
    main()
