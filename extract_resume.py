import json, os, re
 
def store_data(id): 
    with open("resume.txt","r") as file:
        text = file.read()
        data = split_into_units_and_skills(text)
    with open("jd_experience.txt","r") as file:
        lines = []
        for line in file:
            lines.append(line.strip())

    canonical_skills = [] 
    for skill in open("jd_skills.txt", "r"):
        canonical_skills.append(skill.strip())
    print(canonical_skills)

    os.makedirs(f"artifacts/{id}", exist_ok=True)
    write_jsonl_runits(f"artifacts/{id}/resume_units.jsonl",data[0])
    write_jsonl_skills(f"artifacts/{id}/declared_skills.jsonl",data[1])
    write_jsonl_skills(f"artifacts/{id}/canonical_skills.jsonl", canonical_skills)
    write_jsonl_jd(f"artifacts/{id}/jd_lines.jsonl",lines)
    save_manifest(f"artifacts/{id}","resume.txt",f"artifacts/{id}/resume_units.jsonl",f"artifacts/{id}/declared_skills.jsonl","jd_experience.txt",f"artifacts/{id}/jd_lines.jsonl", f"artifacts/{id}/canonical_skills.jsonl")
    
def write_jsonl_runits(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_jsonl_skills(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for skill in data:
            f.write(json.dumps({"type": "skill","text": skill}) + "\n")

def write_jsonl_jd(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps({"type": "jd", "text": line}, ensure_ascii=False) + "\n")

def save_manifest(run_path, resume_text_file, resume_units_file, declared_skills_file, jd_text_file, jd_lines_file, jd_skills_file):
    manifest = {
        "run_id": os.path.basename(run_path),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "rubric": {
            "alignment": 0.40,
            "skills_evidence": 0.30,
            "impact": 0.20,
            "structure_format": 0.10,
        },
        "artifact_version": 1,
        "paths": {
            "resume_text": resume_text_file,
            "resume_units": resume_units_file,
            "declared_skills": declared_skills_file,
            "jd_text": jd_text_file,
            "jd_lines": jd_lines_file,
            "jd_skills": jd_skills_file
        }
    }

    manifest_path = os.path.join(run_path, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

STRONG_VERBS = {
    "built","developed","designed","implemented","deployed","optimized",
    "scaled","automated","refactored","integrated","led","architected",
    "improved","reduced","increased","configured","debugged","launched",
    "migrated","analyzed","tested","maintained","created"
}

# basically recognize patterns n stuff for reading resume so you can store its sections
METRIC_REGEX = re.compile(r"\b\d+(\.\d+)?\s?(%|x|k|m|b)?\b", re.IGNORECASE)

SECTION_HEADERS = {
    "experience": re.compile(r"(?i)^experience\b"),
    "projects": re.compile(r"(?i)^projects?\b"),
    "education": re.compile(r"(?i)^education\b"),
    "skills": re.compile(r"(?i)^(skills|technical skills?|technologies|tech\s*stack|tools|core competencies?)\b"),
    "summary": re.compile(r"(?i)^(summary|profile|about)\b"),
}

SKILL_SPLIT = re.compile(r"[,\|/;•]")

def _is_bullet(line: str) -> bool:
    return bool(re.match(r"^(\*|-|•|\d+[.)])\s+", line))

def _strip_bullet_marker(line: str) -> str:
    return re.sub(r"^(\*|-|•|\d+[.)])\s+", "", line).strip()

def _normalize_skill_token(tok: str) -> str:
    t = tok.strip().lower()
    # remove common qualifiers
    t = re.sub(r"\b(proficient|advanced|intermediate|beginner|familiar)\b", "", t)
    t = re.sub(r"\b(in|with|using)\b", "", t)
    # compact whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def split_into_units_and_skills(text: str):
    """
    Returns:
      units: list of {text, section, strong_verb, has_metric, length_words}
      declared_skills: deduplicated list of skills from the skills/tech section
    """
    units = []
    declared_skills = []

    current_section = None
    capture_skills = False
    
    lines = text.splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Section detection
        for name, pat in SECTION_HEADERS.items():
            if pat.match(line):
                current_section = name
                capture_skills = (name == "skills")
                # If the header line itself has items (e.g., "Skills: Python, SQL")
                header_body = re.sub(pat, "", line, count=1).strip(":—- \t")
                if capture_skills and header_body:
                    for tok in SKILL_SPLIT.split(header_body):
                        tok = _normalize_skill_token(tok)
                        if tok:
                            declared_skills.append(tok)
                # Move to next line
                line = ""
                break
        if not line:
            continue

        # If we are inside the skills section: accumulate skills, not bullets
        if capture_skills:
            # Support bullets or inline lists
            content = _strip_bullet_marker(line) if _is_bullet(line) else line
            for tok in SKILL_SPLIT.split(content):
                tok = _normalize_skill_token(tok)
                if tok:
                    declared_skills.append(tok)
            continue

        # Otherwise, treat as bullet/sentence unit
        bullet = _strip_bullet_marker(line) if _is_bullet(line) else line
        words = bullet.split()
        length_words = len(words)
        first_word = words[0].lower() if words else ""

        units.append({
            "text": bullet,
            "section": current_section,
            "strong_verb": first_word in STRONG_VERBS,
            "has_metric": bool(METRIC_REGEX.search(bullet)),
            "length_words": length_words
        })

    # De-duplicate skills, preserve order
    seen, uniq_skills = set(), []
    for s in declared_skills:
        if s and s not in seen:
            seen.add(s); uniq_skills.append(s)

    return units, uniq_skills

if __name__ == "__main__":
    store_data("5")