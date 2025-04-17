```markdown
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
```

*(Make sure you have an `OPENROUTER_API_KEY` environment variable set before running.)*

## Usage

```bash
python fb_graph.py
```

The script will run `ITERATIONS` cycles:  
1. Append one random suggestion and one random opinion.  
2. Re‑cluster and redraw two side‑by‑side graphs (“Suggestions” & “Opinions”) in candy colors.  
3. Print progress to the console.

To integrate your own feedback streams, replace the `suggestion_pool` and `opinion_pool` lists or modify the feedback‑appending logic in `main()`.

## Configuration

Edit these constants at the top of `fb_graph.py` as needed:

```python
EMBED_MODEL   = "all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.4
ITERATIONS    = 20
K_MIN         = 2
K_MAX_LIMIT   = 10
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
```

## MIT License

```
MIT License

Copyright (c) 2025 Pengwei Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
```
