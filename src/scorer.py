import re
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .parser import extract_skills, extract_experience_years, extract_education


WEIGHTS = {
    "semantic_similarity": 0.50,
    "skill_match": 0.35,
    "experience_match": 0.10,
    "education_match": 0.05,
}


def _skill_score(jd_skills, candidate_skills):
    if not jd_skills:
        return 1.0
    jd = set(jd_skills)
    candidate = set(candidate_skills)
    return len(jd & candidate) / len(jd)


def _experience_score(jd_text, candidate_years):
    required = extract_experience_years(jd_text)
    if required <= 0:
        return 1.0
    return min(candidate_years / required, 1.0)


def _education_score(jd_text, candidate_education):
    required = set(extract_education(jd_text))
    if not required:
        return 1.0
    found = set(candidate_education)
    return 1.0 if required & found else 0.0


def _semantic_scores(jd_text, candidates):
    docs = [jd_text] + [c["text"] for c in candidates]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(docs)
    return cosine_similarity(matrix[0:1], matrix[1:]).flatten()


def rank_candidates(jd_text: str, candidates: List[Dict]) -> List[Dict]:
    jd_skills = extract_skills(jd_text)
    semantic = _semantic_scores(jd_text, candidates)

    rows = []
    for idx, candidate in enumerate(candidates):
        skill = _skill_score(jd_skills, candidate["skills"])
        exp = _experience_score(jd_text, candidate["experience_years"])
        edu = _education_score(jd_text, candidate["education"])
        sem = float(semantic[idx])

        final = 100 * (
            WEIGHTS["semantic_similarity"] * sem
            + WEIGHTS["skill_match"] * skill
            + WEIGHTS["experience_match"] * exp
            + WEIGHTS["education_match"] * edu
        )

        matched = sorted(set(jd_skills) & set(candidate["skills"]))
        missing = sorted(set(jd_skills) - set(candidate["skills"]))

        if matched:
            match_text = ", ".join(matched[:6])
            reason = f"Strongest evidence: matched skills [{match_text}]."
        else:
            reason = "No exact skills from the JD skill dictionary were detected."

        if missing:
            reason += f" Main gaps: {', '.join(missing[:4])}."
        reason += (
            f" TF-IDF similarity={sem:.2f}, skill coverage={skill:.0%}, "
            f"experience match={exp:.0%}, education match={edu:.0%}."
        )

        rows.append({
            "candidate_name": candidate["candidate_name"],
            "file": candidate["file"],
            "final_score": round(final, 1),
            "semantic_similarity": round(sem * 100, 1),
            "skill_match": round(skill * 100, 1),
            "experience_match": round(exp * 100, 1),
            "education_match": round(edu * 100, 1),
            "matched_skills": matched,
            "missing_skills": missing,
            "experience_years_detected": candidate["experience_years"],
            "reasoning": reason,
        })

    rows.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
