import os
import json
from flask import Flask, render_template, request, redirect, url_for

from extract_resume import store_data
from build_vector_db import build_db
from score import score_resume

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "artifacts"


# ---------------------------------------
# Helper: ensure folders exist
# ---------------------------------------
def ensure_run_folder(run_id):
    base = os.path.join(app.config['UPLOAD_FOLDER'], run_id)
    os.makedirs(base, exist_ok=True)
    return base

# ---------------------------------------
# ROUTE: Home Page
# ---------------------------------------
@app.route("/")
def index():
    # List existing run IDs
    path = app.config["UPLOAD_FOLDER"]
    if not os.path.exists(path):
        os.makedirs(path)

    run_ids = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    return render_template("index.html",run_ids=run_ids)

# ---------------------------------------
# ROUTE: Form Submit
# ---------------------------------------
@app.route("/submit", methods=["POST"])
def submit():
    print("Form submitted.")

    # ---------- RUN ID ----------
    existing_id = request.form.get("existing_run_id")
    print("Existing ID: " + str(existing_id))
    new_id = request.form.get("new_run_id").strip()

    
    if new_id:
        run_id = new_id
        print("Using new run ID: " + run_id)
    elif existing_id:
        print("Using existing run ID: " + existing_id)
        run_id = existing_id
        run_folder = ensure_run_folder(run_id)
        results_path = os.path.join(run_folder, "final_score.json")
        result = {}
        with open(results_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        return render_template("results.html",run_id = result["run_id"], result=result)
    else:
        return "Error: You must specify a Run ID.", 400

    # Ensure run folder exists
    run_folder = ensure_run_folder(run_id)

    manifest_path = os.path.join(run_folder, "manifest.json")

    # ---------- FILE UPLOADS ----------

    resume_file = request.files["resume_file"]
    jd_exp_file = request.files["jd_experience_file"]
    jd_skills_file = request.files["jd_skills_file"]

    # Save all uploaded files
    resume_path = os.path.join(run_folder, "resume.txt")
    exp_path = os.path.join(run_folder, "jd_experience.txt")
    skills_path = os.path.join(run_folder, "jd_skills.txt")

    resume_file.save(resume_path)
    jd_exp_file.save(exp_path)
    jd_skills_file.save(skills_path)

    # ---------- RUBRIC ----------
    rubric = {
        "alignment": float(request.form["rubric_alignment"]),
        "skills_evidence": float(request.form["rubric_skills"]),
        "format": float(request.form["rubric_format"]),
    }

    # ---------- HARSHNESS ----------
    harshness = int(request.form["harshness"])

    # ---------- WRITE MANIFEST ----------
    manifest_data = {
        "run_id": run_id,
        "resume_path": resume_path,
        "jd_experience_path": exp_path,
        "jd_skills_path": skills_path,
        "rubric": rubric,
        "harshness": harshness
    }
    

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)

    # ---------- RUN YOUR EXISTING PIPELINE ----------
    # store_data expects run_id only
    store_data(run_id,resume_file,jd_exp_file,jd_skills_file)

    # build_db expects path to manifest.json
    build_db(manifest_path)

    # score_resume expects folder path
    result = score_resume(run_folder)

    return render_template("results.html",run_id = result["run_id"], result=result)


if __name__ == "__main__":
    app.run(debug=True)
