import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import json

def score_resume(run_path):
    manifest = f"{run_path}/manifest.json"
    def best_alignment_for_jd_line(jd_text, k=5):
        q_embed = embed(jd_text)
        res = resume_col.query(query_embeddings=[q_embed],
                                n_results=k,
                                include=["documents","metadatas","distances"]
        )
        matches = top_matches(res, k=k)

        # re-rank by evidence score
        scored = [(evidence_score(meta, sim), sim, doc, meta) for (sim, doc, meta) in matches]
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        return {
            "jd_text": jd_text,
            "resume_text": best[2],
            "cosine_similarity": round(best[1], 3),
            "evidence_score": round(best[0], 3),
            "resume_meta": best[3]
        }
    def best_evidence_for_skill(skill_text, k=5, threshold=0.55):
        q_embed = embed(skill_text)
        res = resume_col.query(query_embeddings=[q_embed], n_results=k, include=["documents","metadatas","distances"])
        matches = top_matches(res, k=k)
        scored = [(evidence_score(meta, sim), sim, doc, meta) for (sim, doc, meta) in matches]
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        demonstrated = best[0] >= threshold
        return {
            "skill": skill_text,
            "demonstrated": demonstrated,
            "resume_text": best[2],
            "cosine_similarity": round(best[1], 3),
            "evidence_score": round(best[0], 3),
            "resume_meta": best[3]
        }
    
    def embed(text:str):
        MODEL_NAME = manifest_data["model"]
        model = SentenceTransformer(MODEL_NAME)
        return model.encode([text],normalize_embeddings=True)[0].tolist()
    
    with open(manifest, "r") as f:
        manifest_data = json.load(f)
    
    
    MODEL = manifest_data["model"]
    THRESHOLD = 0.40
    PATHS = manifest_data["paths"]

    with open(f"{run_path}/manifest.json", "r") as f:
        manifest = json.load(f)
        RUBRIC = manifest["rubric"]
    
    resume_units_path = PATHS["resume_units"]
    jd_lines_path = PATHS["jd_lines"]

    resume_units = read_jsonl(resume_units_path)       
    jd_lines = read_jsonl_text(jd_lines_path)          


    client = chromadb.Client(Settings(is_persistent=True, persist_directory="./vectordb_chroma"))
    
    resume_col = client.get_or_create_collection("resume_sentences", metadata={"hnsw:space": "cosine"})
    skills_canon_col = client.get_or_create_collection("skills_canonical", metadata={"hnsw:space": "cosine"})
    skills_col = client.get_or_create_collection("skills_declared", metadata={"hnsw:space": "cosine"})
    
    dec_skills = skills_col.get()
    dec_skills = dec_skills["documents"]
    can_skills = skills_canon_col.get()
    can_skills = can_skills["documents"]

    
    #QUERY JOB DESCRIPTION
    path = f"{run_path}/jd_alignment.jsonl"
    
    alignments = []
    with open(path, "w", encoding="utf-8") as f:
        index = 0
        for line in jd_lines:
            result = best_alignment_for_jd_line(line)
            result["jd_index"] = index
            alignments.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            index += 1
        
    # FIND MATCHES FOR SKILLS
    matches = []
    
    for skill in can_skills:
        skill_split = skill.split()
        skill = skill_split[2]
        q_embed = embed(skill) # get the actual word, not extended form
        res = skills_col.query(query_embeddings=[q_embed], n_results=1, include=["documents","metadatas","distances"])
        id = res["ids"][0][0] # of closest canonical skill
        res["distances"][0][0] = 1 - res["distances"][0][0] #convert to similarity
        similarity = res["distances"][0][0]

        if similarity >= THRESHOLD:
            matches.append({"canonical_skill": skill, "declared_skill": res["documents"][0][0], "similarity": res["distances"][0][0], "declared_skill_id": id})
        else:
            pass

    # now that we have our matches, we can score them based on evidence in the resume
    for match in matches:
        skill_text = match["canonical_skill"]
        evidence = best_evidence_for_skill(skill_text, k=5, threshold=THRESHOLD)
        match["evidence"] = evidence

    alignment_score = score_alignment(alignments)    
    skills_score = score_skill_evidence(matches, len(can_skills))
    structure_score = score_structure_format(resume_units)

    print(f"Alignment Score: {alignment_score}")
    print(f"Skills Evidence Score: {skills_score}") 
    print(f"Structure & Format Score: {structure_score}")

def score_alignment(alignments, cover_thresh=0.55, w_mean=0.85, w_cov=0.15):
    if not alignments:
        return 0.0

    evs = [float(a.get("evidence_score", 0.0)) for a in alignments]
    mean_ev = sum(evs) / len(evs)

    # Coverage = share of JD bullets with "good enough" evidence
    covered = sum(1 for x in evs if x >= cover_thresh)
    coverage = covered / len(evs)

    return max(0.0, min(1.0, w_mean * mean_ev + w_cov * coverage))


def score_skill_evidence(matches, total_canonical_skills, cover_thresh=0.55, alpha=0.6, gamma=1.0):
    for i, match in enumerate(matches):
        print(f"Match {i+1}: Canonical Skill: {match['canonical_skill']}, Declared Skill: {match['declared_skill']}, Similarity: {match['similarity']}, Evidence Score: {match['evidence']['evidence_score']}\n")
    if not total_canonical_skills or total_canonical_skills <= 0:
        return 0.0

    if not matches:
        return 0.0

    demonstrated = len(matches)
    coverage = ((demonstrated / total_canonical_skills) + 1)/2 # be lighter on coverage

    evs = []
    for m in matches:
        ev = float(m["evidence"]["evidence_score"])
        ev = max(0.0, min(1.0, ev))
        evs.append(ev)
    #avg
    quality = sum(evs) / len(evs)
    quality = (quality+1)/2 # be more generous for it existing, less focus on structure
    print(f"quality: {quality}, coverage: {coverage}")

    #score will be the average between the quality and how much is covered
    score = (quality + coverage) / 2

    return score

_SECTION_ALIASES = {
    "summary": {"summary", "profile"},
    "skills": {"skills", "technical skills", "tech skills"},
    "experience": {"experience", "work experience", "employment"},
    "projects": {"projects"},
    "education": {"education", "academics"},
}

def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))

def _safe_div(a, b, default=0.0):
    return a / b if b else default

def score_structure_format(resume_units):
    """
    Structure: presence/order of key sections, bullets consistency, basic formatting hygiene.
    Requires that each unit has meta like {"section": "experience"} or you supply section labels elsewhere.
    """
    if not resume_units:
        return 0.0

    sections_seen = []
    bullets_like = 0
    total_lines = 0

    for u in resume_units:
        sec = (u.get("section") or "").strip().lower()
        if sec:
            sections_seen.append(sec)
        txt = (u.get("text") or u.get("document") or "").strip()
        if txt:
            total_lines += 1
            if txt.startswith(("-", "•", "*")):
                bullets_like += 1

    # Section coverage across canonical buckets
    seen_norm = set()
    for s in sections_seen:
        for key, aliases in _SECTION_ALIASES.items():
            if s in aliases:
                seen_norm.add(key)

    # Expect at least: summary, skills, experience, education; projects is a plus
    must_have = {"summary", "skills", "experience", "education"}
    base_cov = len(must_have & seen_norm) / len(must_have)
    projects_bonus = 0.05 if "projects" in seen_norm else 0.0

    # Bullet consistency (not strict, just a nudge)
    bullet_rate = _safe_div(bullets_like, total_lines)
    bullet_component = min(0.15, bullet_rate * 0.15 / 0.7)  # 70% bullets ~ max 0.15

    score = _clamp(0.80 * base_cov + projects_bonus + bullet_component)
    return score

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def read_jsonl_text(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_json = json.loads(line)
            data.append(line_json["text"])
    return data

def top_matches(result, k=3):
    # for distances lower is better so, 1 - distance
    out = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        sim = 1.0 - dist
        out.append((sim, doc, meta))
    #put in descending order
    out.sort(key=lambda x: x[0], reverse=True)
    return out[:k]

# this function outputs a score based on the similarity * metadata factors
def evidence_score(meta, sim):
    # soften the similarity scores
    def ramp(sim, lo=0.35, hi=0.75):
        if sim <= lo: return 0.0
        if sim >= hi: return 1.0
        return (sim - lo) / (hi - lo)
    
    bonus = 0.0
    bonus += ramp(sim)

    section = meta.get("section")
    if section == "experience":
        bonus += 0.07
    elif section == "projects":
        bonus += 0.04
    else:
        bonus += 0.00  # leave other sections neutral

    # Style signals (keep small so they help but don’t dominate)
    if meta.get("strong_verb"):
        bonus += 0.03
    if meta.get("has_metric"):
        bonus += 0.06

    # Length nudge (tiny)
    L = int(meta.get("length_words", 0))
    if 6 <= L <= 28:
        bonus += 0.02
    # no penalty for short/long; we avoid harshness

    return min(1.0, sim + bonus)

if __name__ == "__main__":
    score_resume("artifacts/5/")