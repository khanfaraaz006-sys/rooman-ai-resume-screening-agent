import re
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader
from docx import Document


SKILLS = [
    "python", "java", "c++", "javascript", "sql", "mysql", "postgresql",
    "machine learning", "deep learning", "artificial intelligence", "ai",
    "nlp", "natural language processing", "computer vision", "pandas",
    "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras",
    "flask", "fastapi", "django", "streamlit", "git", "github", "docker",
    "aws", "azure", "gcp", "rest api", "api", "llm", "large language model",
    "rag", "retrieval augmented generation", "prompt engineering",
    "langchain", "transformers", "hugging face", "data analysis",
    "data science", "excel", "power bi", "tableau"
]

DEGREE_TERMS = [
    "b.e", "be ", "b.tech", "btech", "bachelor", "m.tech", "mtech",
    "master", "b.sc", "bsc", "m.sc", "msc", "computer science",
    "information technology", "engineering"
]


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_resume(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext == ".txt":
        return _read_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")


def extract_name(text: str, fallback: str) -> str:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        first = re.sub(r"[^A-Za-z .'-]", "", lines[0]).strip()
        if 2 <= len(first) <= 50 and len(first.split()) <= 5:
            return first
    return fallback.replace("_", " ").replace("-", " ").title()


def extract_skills(text: str) -> List[str]:
    lower = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
        if re.search(pattern, lower):
            found.append(skill)
    # Normalize aliases while preserving readable labels
    aliases = {
        "ai": "artificial intelligence",
        "sklearn": "scikit-learn",
        "api": "rest api",
        "large language model": "llm",
        "retrieval augmented generation": "rag",
    }
    normalized = []
    for s in found:
        s = aliases.get(s, s)
        if s not in normalized:
            normalized.append(s)
    return normalized


def extract_experience_years(text: str) -> float:
    lower = text.lower()
    values = []
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|year|yrs|yr)\s+(?:of\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\+?\s*(?:years|year|yrs|yr)",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, lower):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass

    # Fresher / internship-friendly fallback
    if values:
        return max(values)
    if "fresher" in lower:
        return 0.0
    internship_count = len(re.findall(r"\bintern(?:ship)?\b", lower))
    return min(1.0, internship_count * 0.25)


def extract_education(text: str) -> List[str]:
    lower = text.lower()
    return sorted({term for term in DEGREE_TERMS if term in lower})


def parse_resume(path: Path) -> Dict:
    text = read_resume(path)
    return {
        "file": path.name,
        "candidate_name": extract_name(text, path.stem),
        "text": text,
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
    }
