/**
 * CampusPlacementIQ Dashboard Interactive Engine

 * - Live Interactive Placement Simulator (Real-Time ML Inference)
 * - Gallery Category Filtering
 * - Image Modal / Lightbox preview
 * - Sidebar Live Session Search
 * - Monospace Terminal Copy Button
 */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Live Interactive ML Inference Simulator
  const sCGPA = document.getElementById("sliderCGPA");
  const sCoding = document.getElementById("sliderCoding");
  const sAptitude = document.getElementById("sliderAptitude");
  const sInterview = document.getElementById("sliderInterview");
  const sAttendance = document.getElementById("sliderAttendance");
  const sBacklogs = document.getElementById("selectBacklogs");

  const vCGPA = document.getElementById("valCGPA");
  const vCoding = document.getElementById("valCoding");
  const vAptitude = document.getElementById("valAptitude");
  const vInterview = document.getElementById("valInterview");
  const vAttendance = document.getElementById("valAttendance");

  const outProb = document.getElementById("simProbability");
  const outOutcome = document.getElementById("simOutcome");
  const outBadge = document.getElementById("simBadge");
  const outSalary = document.getElementById("simSalary");

  function calculatePlacementLive() {
    if (!sCGPA) return;

    const cgpa = parseFloat(sCGPA.value);
    const coding = parseFloat(sCoding.value);
    const aptitude = parseFloat(sAptitude.value);
    const interview = parseFloat(sInterview.value);
    const attendance = parseFloat(sAttendance.value);
    const backlogs = parseInt(sBacklogs.value, 10);

    // Update value pill labels
    if (vCGPA) vCGPA.innerText = cgpa.toFixed(2);
    if (vCoding) vCoding.innerText = Math.round(coding);
    if (vAptitude) vAptitude.innerText = Math.round(aptitude);
    if (vInterview) vInterview.innerText = Math.round(interview);
    if (vAttendance) vAttendance.innerText = Math.round(attendance) + "%";

    // Instant local calculation using Session 09 Logistic Regression weights
    const cgpa_z = (cgpa - 7.0) / 1.2;
    const coding_z = (coding - 50.0) / 20.0;
    const aptitude_z = (aptitude - 60.0) / 18.0;
    const interview_z = (interview - 55.0) / 20.0;
    const attendance_z = (attendance - 75.0) / 15.0;

    const z = 1.45 * cgpa_z + 1.15 * coding_z + 0.85 * aptitude_z + 0.95 * interview_z + 0.40 * attendance_z - 1.20 * backlogs + 0.65;
    const prob = 1.0 / (1.0 + Math.exp(-z));
    const probPct = (prob * 100).toFixed(1);

    if (outProb) outProb.innerText = probPct + "%";

    if (prob >= 0.50) {
      if (outOutcome) outOutcome.innerText = "Likely Placed (Success)";
      if (outProb) outProb.style.color = "#10b981";
      if (outBadge) {
        if (prob >= 0.80) {
          outBadge.innerText = "High Probability";
          outBadge.style.color = "#10b981";
        } else {
          outBadge.innerText = "Moderate Probability";
          outBadge.style.color = "#06b6d4";
        }
      }
      const estSal = Math.max(3.5, 3.2 + (cgpa - 5.0) * 1.8 + (coding / 100.0) * 8.5).toFixed(1);
      if (outSalary) outSalary.innerText = "₹" + estSal + " LPA";
    } else {
      if (outOutcome) outOutcome.innerText = "At-Risk Profile";
      if (outProb) outProb.style.color = "#f43f5e";
      if (outBadge) {
        outBadge.innerText = "Needs Improvement";
        outBadge.style.color = "#f43f5e";
      }
      if (outSalary) outSalary.innerText = "Unplaced (₹0 LPA)";
    }
  }

  if (sCGPA) {
    [sCGPA, sCoding, sAptitude, sInterview, sAttendance].forEach((slider) => {
      slider.addEventListener("input", calculatePlacementLive);
    });
    if (sBacklogs) sBacklogs.addEventListener("change", calculatePlacementLive);
    calculatePlacementLive();
  }

  // 2. Sidebar Session Search Filter
  const searchInput = document.getElementById("sidebarSearch");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      const query = e.target.value.toLowerCase().trim();
      const sessionLinks = document.querySelectorAll(".session-nav-item");
      const moduleGroups = document.querySelectorAll(".module-group");

      sessionLinks.forEach((link) => {
        const title = link.getAttribute("data-title") || link.innerText.toLowerCase();
        const matches = title.includes(query);
        link.style.display = matches ? "flex" : "none";
      });

      moduleGroups.forEach((group) => {
        const visibleSessions = group.querySelectorAll(".session-nav-item[style='display: flex;']");
        const allHidden = group.querySelectorAll(".session-nav-item[style='display: none;']");
        if (query && visibleSessions.length === 0 && allHidden.length > 0) {
          group.style.display = "none";
        } else {
          group.style.display = "block";
        }
      });
    });
  }

  // 3. Gallery Category Filtering
  const filterBtns = document.querySelectorAll(".filter-pill-btn");
  const galleryCards = document.querySelectorAll(".showcase-card");

  if (filterBtns.length > 0) {
    filterBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        filterBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");

        const category = btn.getAttribute("data-category");

        galleryCards.forEach((card) => {
          const cardCat = card.getAttribute("data-category") || "all";
          if (category === "all" || cardCat.includes(category)) {
            card.style.display = "flex";
          } else {
            card.style.display = "none";
          }
        });
      });
    });
  }

  // 4. Image Modal (Lightbox)
  const modal = document.getElementById("imageLightboxModal");
  const modalImg = document.getElementById("lightboxImage");
  const modalTitle = document.getElementById("lightboxTitle");
  const modalSub = document.getElementById("lightboxSub");
  const closeModalBtn = document.getElementById("lightboxClose");

  const clickableImages = document.querySelectorAll(".showcase-image-holder, .session-plot-img-wrap");

  clickableImages.forEach((wrap) => {
    wrap.style.cursor = "zoom-in";
    wrap.addEventListener("click", () => {
      const img = wrap.querySelector("img");
      if (!img || !modal) return;

      const card = wrap.closest(".showcase-card, .session-plot-card");
      const titleText = card ? (card.querySelector(".showcase-plot-title, .plot-name-text")?.innerText || img.alt) : img.alt;
      const subText = card ? (card.querySelector(".session-badge-chip, .plot-path-sub")?.innerText || "") : "";

      modalImg.src = img.src;
      modalTitle.innerText = titleText;
      modalSub.innerText = subText;
      modal.classList.add("active");
      document.body.style.overflow = "hidden";
    });
  });

  if (closeModalBtn) {
    closeModalBtn.addEventListener("click", closeModal);
  }
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });
  }

  function closeModal() {
    if (modal) {
      modal.classList.remove("active");
      document.body.style.overflow = "";
    }
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal && modal.classList.contains("active")) {
      closeModal();
    }
  });

  // 5. Copy Terminal Report Output
  const copyBtn = document.getElementById("copyReportBtn");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const codeBlock = document.querySelector(".console-text");
      if (codeBlock) {
        navigator.clipboard.writeText(codeBlock.innerText).then(() => {
          const originalHTML = copyBtn.innerHTML;
          copyBtn.innerHTML = "<span>✓</span> Copied!";
          copyBtn.style.background = "#10b981";
          copyBtn.style.color = "#ffffff";
          setTimeout(() => {
            copyBtn.innerHTML = originalHTML;
            copyBtn.style.background = "";
            copyBtn.style.color = "";
          }, 2000);
        });
      }
    });
  }
});
