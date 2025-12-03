from extract_resume import store_data
from build_vector_db import build_db
from score import score_resume

def main():
    
    run_id = "2"
    store_data(run_id)
    build_db(f"artifacts/{run_id}/manifest.json")
    score_resume(f"artifacts/{run_id}/")

if __name__ == "__main__":
    main()