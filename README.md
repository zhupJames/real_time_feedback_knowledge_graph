# Real-Time Feedback Knowledge Graph

A Python tool that dynamically clusters user feedback, extracts core keywords, and visualizes them as a color‑coded knowledge graph in real time.

## Features

- **Dynamic clustering**  
  Automatically chooses the optimal number of clusters (between `K_MIN` and `K_MAX_LIMIT`) based on silhouette score; handles small datasets (<4 items) as individual clusters.  
- **Keyword extraction**  
  Calls Openrouter (Deepseek) to extract one representative keyword per cluster, with a fallback to the first token on failure.  
- **Candy‑style coloring**  
  Projects cluster centers to 1D and maps them to a Pastel1 (“candy”) gradient—closer clusters get more similar colors.  
- **Interactive graph**  
  Node size ∝ cluster size; edges drawn for cosine similarity ≥ `SIM_THRESHOLD`; updates live over multiple iterations.

## Requirements

- Python 3.7+  
- [sentence-transformers](https://github.com/sentence-transformers/sentence-transformers)  
- `scikit-learn`  
- `networkx`  
- `matplotlib`  
- `requests`

## Installation

```bash
git clone https://github.com/your‑repo/real‑time‑feedback‑kg.git
cd real‑time‑feedback‑kg
pip install -r requirements.txt
