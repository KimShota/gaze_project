# Gaze Project  

This repository contains research scripts and datasets for analyzing **gaze behavior** and its relationship to **attention, immersion, and mental fatigue**. It builds on large-scale VR eye-tracking experiments and provides tools for generating **heatmaps, frequency/duration metrics, and statistical comparisons** across different image-based tasks.  

The project supports analysis of **gaze distribution across 15 image pairs (A–O)**, enabling comparisons between **left vs right images** and different orientations (portrait vs landscape).  

---

## 📂 Repository Structure  

gaze_project/
├── Images/ # Base image assets used for overlays
├── Pair_A/ # Gaze data and heatmaps for image pair A
├── Pair_B/ # Gaze data and heatmaps for image pair B
├── Pair_C/
├── Pair_D/
├── Pair_E/
├── Pair_F/
├── Pair_G/
├── Pair_H/
├── Pair_I/
├── Pair_J/
├── Pair_K/
├── Pair_L/
├── Pair_M/
├── Pair_N/
├── Pair_O/ # Gaze data and heatmaps for image pair O
│
├── gaze_analysis_batcher.py # Batch processing of gaze data & heatmap generation
└── README.md # Project documentation


---

## 🔬 Key Features  

- **Gaze Heatmap Generation**  
  - Frequency mode → counts non-consecutive visits to each cell.  
  - Duration mode → calculates fixation time on each grid cell.  
  - Outputs normalized percentages (0–100%) across participants.  

- **Support for 15 Image Pairs (A–O)**  
  - Each pair includes **left (1) and right (2)** images.  
  - Categories include **Portrait Left, Portrait Right, Landscape Left, Landscape Right**.  

- **Cell-Based Analysis**  
  - Each image is divided into a **16×16 grid (256 cells)**.  
  - Gaze hits are mapped to `cellID` values (1–1024 depending on orientation).  

- **Batch Processing**  
  - `gaze_analysis_batcher.py` automates reading multiple CSVs.  
  - Aggregates gaze data across participants for group-level heatmaps.  

- **Comparative Analysis**  
  - Supports difference heatmaps between groups (e.g., high vs low fatigue).  
  - Calculates p-values for statistical significance at the grid-cell level.  

---

## 📊 Workflow  

1. **Input Data**  
   - Gaze tracking CSVs (per participant).  
   - Includes timestamp, `cellID`, and fixation duration.  

2. **Processing**  
   - Normalize participant data.  
   - Map gaze hits to grids (Portrait or Landscape).  
   - Apply frequency/duration calculations.  

3. **Outputs**  
   - Heatmaps per image pair (A–O).  
   - Difference heatmaps (between participant groups).  
   - CSV summaries of normalized attention percentages.  

---

## 📈 Research Context  

This repository is part of a larger **VR-based mental fatigue research system**, focused on:  

- **Analyzing gaze distribution patterns** to assess attention and fatigue.  
- **Comparing groups** (high vs low fatigue participants).  
- **Building difference heatmaps** to highlight cognitive fatigue markers.  
- **Statistical validation** using p-values per grid cell.  

Findings so far:  
- High fatigue participants show **less consistent gaze coverage** and more concentrated fixations.  
- Low fatigue participants demonstrate **wider exploration** and shorter fixation durations.  
- Difference heatmaps reveal **systematic shifts** in gaze between groups.  

---

## 🛠 Tech Stack  

- **Python**  
  - pandas  
  - NumPy  
  - matplotlib  
  - seaborn  
  - statsmodels  
  - scikit-learn  

---

## 🚀 Usage  

Clone the repository:  
```bash
git clone https://github.com/KimShota/gaze_project.git
cd gaze_project
Run the batch analysis:

python gaze_analysis_batcher.py


Results will be saved in the corresponding Pair folders (A–O) with heatmaps and CSV outputs.
