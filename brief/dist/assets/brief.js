"use strict";

fetch("brief.json")
  .then((response) => {
    if (!response.ok) throw new Error("brief data unavailable");
    return response.json();
  })
  .then((data) => {
    document.querySelector('[data-metric="base"]').textContent = data.result.base_exact_percent.toFixed(2) + "%";
    document.querySelector('[data-metric="lora"]').textContent = data.result.lora_exact_percent.toFixed(2) + "%";
    document.getElementById("data-status").textContent = "Evidence loaded";
  })
  .catch(() => {
    document.getElementById("data-status").textContent = "Static evidence";
  });
