from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import hashlib
import json

def build_db(manifest):
    with open(manifest, "r") as f:
        manifest_data = json.load(f)

    MODEL_NAME = manifest_data["model"]
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.Client(Settings(is_persistent=True, persist_directory="./vectordb_chroma"))

    existing_collections = client.list_collections()
    print(existing_collections)

    # Check if the collection exists
    for collection_name in existing_collections:
        print(str(collection_name.name))
        if any(c.name == collection_name.name for c in existing_collections):
            print("deleting: " + str(collection_name.name))
            client.delete_collection(name=collection_name.name)
        else:
            print("No collection with name: " + str(collection_name.name))
 

    resume_col = client.get_or_create_collection("resume_sentences", metadata={"hnsw:space": "cosine"})
    jd_col = client.get_or_create_collection("jd_requirements",  metadata={"hnsw:space": "cosine"})
    skills_col = client.get_or_create_collection("skills_declared", metadata={"hnsw:space": "cosine"})
    skills_canon_col = client.get_or_create_collection("skills_canonical", metadata={"hnsw:space": "cosine"})

    def embed(text:str):
        return model.encode([text],normalize_embeddings=True)[0].tolist()

    def make_embedding_id(type,text: str, index: int) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{type}::{h}-{index}"
    
    resume_units_path = manifest_data["paths"]["resume_units"]
    declared_skills_path = manifest_data["paths"]["declared_skills"]
    jd_lines_path = manifest_data["paths"]["jd_lines"]
    canonical_skills_path = manifest_data["paths"]["jd_skills"]
    run_id = manifest_data["run_id"]

    i = 0
    for unit in open(resume_units_path):
        obj = json.loads(unit)
        resume_bullet = obj["text"]
        resume_meta = {
            "section": obj["section"],
            "strong_verb": obj["strong_verb"],
            "has_metric": obj["has_metric"],
            "length_words": obj["length_words"],
            "index": i
        }

        emb_id = make_embedding_id("resume",resume_bullet, i)
        resume_col.upsert(
            ids=[emb_id],
            embeddings=[embed(resume_bullet)],
            documents=[resume_bullet],
            metadatas=[resume_meta]
        )
        i += 1

    # single word embeddings are not as meaningful
    def expand_skill(skill):
        return f"Evidence of {skill} demonstrated through real-world projects, responsibilities, or achievements."

    i = 0
    for skill in open(declared_skills_path):
        obj = json.loads(skill)
        skill_name = obj["text"]
        emb_id = make_embedding_id("skill",skill_name, i)
        skills_col.upsert(
            ids=[emb_id],
            embeddings=[embed(skill_name)],  
            documents=[skill_name],
            metadatas=[{"name": skill_name}])
        i += 1

    i = 0
    for skill in open(canonical_skills_path):
        obj = json.loads(skill)
        skill_name = obj["text"]
        skill_name = expand_skill(skill_name)
        emb_id = make_embedding_id("skill",skill_name, i)
        skills_canon_col.upsert(
            ids=[emb_id],
            embeddings=[embed(skill_name)],  
            documents=[skill_name],
            metadatas=[{"name": skill_name}])
        i += 1
    
    i = 0
    for jd_line in open(jd_lines_path):
        obj = json.loads(unit)
        jd_text = obj["text"]

        emb_id = make_embedding_id("jd",jd_text, i)
        jd_col.upsert(
            ids=[emb_id],
            embeddings=[embed(jd_text)],
            documents=[jd_line],
            metadatas=[{"index": 2}]
        )
        i += 1
    

        
    
if __name__ == "__main__":
    #build_db()
    build_db("artifacts/5/manifest.json")
