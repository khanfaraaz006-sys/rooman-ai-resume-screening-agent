from src.parser import extract_skills, extract_experience_years
from src.scorer import rank_candidates


def test_extract_skills():
    skills = extract_skills("Python, machine learning, SQL and Git")
    assert "python" in skills
    assert "machine learning" in skills
    assert "sql" in skills
    assert "git" in skills


def test_rank_order():
    jd = "Need Python machine learning NLP SQL Git. Bachelor's Computer Science. Fresher welcome."
    candidates = [
        {
            "file": "strong.txt",
            "candidate_name": "Strong",
            "text": "B.Tech Computer Science Python machine learning NLP SQL Git fresher",
            "skills": extract_skills("Python machine learning NLP SQL Git"),
            "experience_years": 0.0,
            "education": ["b.tech", "computer science"],
        },
        {
            "file": "weak.txt",
            "candidate_name": "Weak",
            "text": "Java frontend React",
            "skills": extract_skills("Java frontend React"),
            "experience_years": 0.0,
            "education": [],
        },
    ]
    ranked = rank_candidates(jd, candidates)
    assert ranked[0]["candidate_name"] == "Strong"
    assert ranked[0]["final_score"] > ranked[1]["final_score"]
