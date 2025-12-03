# file: chroma_mvp.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

#CURRENTLY AN EXAMPLE
# -------------------------
# 0) Setup: model + client
# -------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

def embed(text: str):
    # normalize_embeddings=True gives unit vectors → cosine similarity via (1 - distance)
    return model.encode([text], normalize_embeddings=True)[0].tolist()

client = chromadb.Client(Settings(is_persistent=True, persist_directory="./vectordb_chroma"))

resume_col = client.get_or_create_collection("resume_sentences", metadata={"hnsw:space": "cosine"})
jd_col     = client.get_or_create_collection("jd_requirements",  metadata={"hnsw:space": "cosine"})
skills_col = client.get_or_create_collection("skills_canonical", metadata={"hnsw:space": "cosine"})

# -----------------------------------------
# 1) Upsert a resume bullet + JD line + skill
# -----------------------------------------
resume_bullet = "Deployed microservices to AWS EKS using Docker and Kubernetes; reduced deployment time by 35%."
resume_meta = {
    "section": "experience",
    "strong_verb": True,
    "has_metric": True,
    "length_words": len(resume_bullet.split()),
    "index": 12
}
resume_col.upsert(
    ids=["R::12"],
    embeddings=[embed(resume_bullet)],
    documents=[resume_bullet],
    metadatas=[resume_meta]
)

jd_line = "Deploy containerized applications using Kubernetes."
jd_col.upsert(
    ids=["J::2"],
    embeddings=[embed(jd_line)],    # optional; could re-embed at query time
    documents=[jd_line],
    metadatas=[{"index": 2}]
)

skill_name = "typescript"
skills_col.upsert(
    ids=["S::typescript"],
    embeddings=[embed(skill_name)],  # optional; could re-embed at query time
    documents=[skill_name],
    metadatas=[{"name": "typescript"}]
)

# ------------------------------------------------
# 2) Helper: convert Chroma result to (sim, doc, meta)
# ------------------------------------------------
def top_matches(result, k=3):
    # result["distances"] -> lower is better (cosine distance); sim = 1 - distance
    out = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        sim = 1.0 - dist
        out.append((sim, doc, meta))
    out.sort(key=lambda x: x[0], reverse=True)
    return out[:k]

def evidence_score(meta, sim):
    # simple quality weighting using metadata
    section_w = 1.00 if meta.get("section") == "experience" else (0.80 if meta.get("section") == "projects" else 0.50)
    verb_m    = 1.10 if meta.get("strong_verb") else 1.00
    metric_m  = 1.20 if meta.get("has_metric") else 1.00
    L         = int(meta.get("length_words", 0))
    length_m  = 1.05 if 6 <= L <= 28 else (0.75 if L < 6 else 0.85)
    return sim * section_w * verb_m * metric_m * length_m

# ------------------------------------------
# 3) JD → Resume: alignment (query-by-text)
# ------------------------------------------
def best_alignment_for_jd_line(jd_text, k=5):
    q_embed = embed(jd_text)
    res = resume_col.query(query_embeddings=[q_embed], n_results=k, include=["documents","metadatas","distances"])
    matches = top_matches(res, k=k)
    # re-rank by evidence score
    scored = [(evidence_score(meta, sim), sim, doc, meta) for (sim, doc, meta) in matches]
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[1]
    return {
        "jd_text": jd_text,
        "resume_text": best[2],
        "cosine_similarity": round(best[1], 3),
        "evidence_score": round(best[0], 3),
        "resume_meta": best[3]
    }

# ------------------------------------------
# 4) Skill → Resume: demonstrated evidence
# ------------------------------------------
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

# ------------------------------------------
# 5) Demo calls
# ------------------------------------------
if __name__ == "__main__":
    print("=== JD → Resume (Alignment) ===")
    print(best_alignment_for_jd_line(jd_line))

    print("\n=== Skill → Resume (Evidence) ===")
    print(best_evidence_for_skill("typescript"))
