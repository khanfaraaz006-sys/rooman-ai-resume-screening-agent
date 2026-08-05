import argparse
import json
from pathlib import Path

from src.parser import parse_resume
from src.scorer import rank_candidates
from src.utils import load_text_file, save_csv, save_json


def main():
    parser = argparse.ArgumentParser(description="AI Resume Screening Agent")
    parser.add_argument("--jd", required=True, help="Path to job description text file")
    parser.add_argument("--resumes", required=True, help="Folder containing PDF, DOCX, or TXT resumes")
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()

    jd_text = load_text_file(args.jd)
    resume_dir = Path(args.resumes)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    supported = {".pdf", ".docx", ".txt"}
    candidates = []
    for path in sorted(resume_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in supported:
            try:
                candidates.append(parse_resume(path))
            except Exception as exc:
                print(f"[WARN] Skipping {path.name}: {exc}")

    if not candidates:
        raise SystemExit("No readable resumes found.")

    ranked = rank_candidates(jd_text, candidates)
    save_json(ranked, out_dir / "ranked_candidates.json")
    save_csv(ranked, out_dir / "ranked_candidates.csv")

    print("\n=== RANKED CANDIDATES ===")
    for row in ranked:
        print(
            f"{row['rank']:>2}. {row['candidate_name']:<20} "
            f"Score: {row['final_score']:>5.1f}/100 | {row['file']}"
        )
        print(f"    {row['reasoning']}")
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
