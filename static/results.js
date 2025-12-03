document.addEventListener("DOMContentLoaded", () => {
    const fills = document.querySelectorAll(".score-bar-fill[data-score]");

    fills.forEach(el => {
        const raw = el.getAttribute("data-score");
        const value = parseFloat(raw);

        if (!isNaN(value)) {
            const clamped = Math.max(0, Math.min(100, value));
            el.style.width = clamped + "%";
        } else {
            el.style.width = "0%";
        }
    });
});
