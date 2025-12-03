import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import json
import os

def main():
    # Initialize ChromaDB client
    client = chromadb.Client(Settings(is_persistent=True, persist_directory="./vectordb_chroma"))

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    resume_col = client.get_or_create_collection("resume_sentences", metadata={"hnsw:space": "cosine"})
    jd_col = client.get_or_create_collection("jd_requirements",  metadata={"hnsw:space": "cosine"})
    skills_col = client.get_or_create_collection("skills_declared", metadata={"hnsw:space": "cosine"})
    skills_canon_col = client.get_or_create_collection("skills_canonical", metadata={"hnsw:space": "cosine"})

    dec_skills = skills_col.get()
    dec_skills = dec_skills["documents"]
    can_skills = skills_canon_col.get()
    can_skills = can_skills["documents"]
    matches = []
    THRESHOLD = .40

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
    print(matches)

def embed(text:str):
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(MODEL_NAME)
    return model.encode([text],normalize_embeddings=True)[0].tolist()

if __name__ == "__main__":
    main()