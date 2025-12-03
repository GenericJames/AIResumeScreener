document.addEventListener("DOMContentLoaded", () => {

    const form = document.querySelector("form");
    const waitMsg = document.getElementById("please-wait");

    const showBtn = document.getElementById("create_run_btn");
    const file_upload_section = document.getElementById("file-upload-section");
    const existing_section = document.getElementById("existing_resume_section");
    const add_new_section = document.getElementById("add_new_run_section");

    const existingSelect = document.getElementById("existing_run_id");
    const newRunInput = document.getElementById("new_run_id");

    const resumeFile = document.getElementById("resume_file");
    const jdExpFile = document.getElementById("jd_experience_file");
    const jdSkillsFile = document.getElementById("jd_skills_file");

    const fileInputs = [resumeFile, jdExpFile, jdSkillsFile];

    /* ============================
       SHOW/HIDE NEW RUN SECTIONS
       ============================ */
    showBtn.addEventListener("click", () => {
        file_upload_section.style.display = "block";
        existing_section.style.display = "none";
        add_new_section.style.display = "block";
    });

    /* ============================
       FILE REQUIREMENTS LOGIC
       ============================ */

    function updateRequirements() {
        const usingExisting = existingSelect.value && !newRunInput.value.trim();

        // If using an existing run, files are NOT required
        // If creating a new run, files ARE required
        fileInputs.forEach(input => {
            input.required = !usingExisting;
        });
    }

    existingSelect.addEventListener("change", updateRequirements);
    newRunInput.addEventListener("input", updateRequirements);

    // Initial call so the state is correct on page load
    updateRequirements();

    /* ============================
       SUBMIT MESSAGE
       ============================ */
    form.addEventListener("submit", () => {
        if (waitMsg) waitMsg.style.display = "block";
    });
});

