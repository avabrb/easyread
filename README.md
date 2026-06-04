# AMR-GNN Easy Read Text Simplification Pipeline

An end-to-end NLP pipeline that simplifies text into the "Easy Read" format (designed for accessibility for people with intellectual disabilities) by using Abstract Meaning Representation (AMR) graphs, Graph Neural Networks (GNNs), and Large Language Models (LLMs).

This project integrates structural GNN predictions over semantic graphs with LLM generation prompts to produce simplified, fluent, and conceptually preserved Easy Read segments.

---

## System Architecture

The pipeline consists of the following phases:
1. **Segmentation**: Splitting the input sentences based on `_seg_` markers.
2. **LLM Rewriting (Claude)**: Completing fractured segments into standalone, simple sentences via the Claude API.
3. **AMR Graph Parsing**: Constructing Abstract Meaning Representation (AMR) graphs from normal sentences and easy segments.
4. **GNN Training**: Encoding the semantic graphs and predicting concept importance and structural linkages.
5. **LLM Synthesis & Evaluation**: Generating final Easy Read texts guided by GNN outputs and calculating evaluation metrics (SARI, BLEU, FK Readability).

---

## Directory Structure

```text
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── raw-data/                       # Source segmentation files (.seg)
│   ├── en/
│   ├── es/
│   └── eu/
├── data-preprocessing/
│   ├── amr-generation.ipynb        # Jupyter Notebook for AMR generation
│   ├── amr_generation.py           # Standalone CLI script for AMR generation
│   ├── segmentation.py             # Parses source data into initial segments
│   ├── rewrite_easy_segments.py    # Standalone LLM sentence completion script
│   ├── post-processing.py          # Flags incomplete sentences and merges manual fixes
│   └── processed-data/             # Outputs of preprocessing pipeline steps (.jsonl)
├── training/
│   ├── gnn-training.ipynb          # Jupyter Notebook for GNN training & synthesis
│   ├── gnn_training.py             # Standalone CLI script for GNN training & synthesis
│   └── intermediate-results/       # Intermediate results on a subset of data
└── results/
    ├── paper_statistics.json       # Overall evaluation results
    ├── evaluation_scores_all.json
    ├── plots_all/                  # Plots comparing GNN+LLM output with baselines
    └── plots_limitedscope/         # Plots for subset datasets
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies
Make sure you have Python 3.10+ installed. Install the Python packages:
```bash
pip install -r requirements.txt
```

### 3. Install spaCy NLP Models
Download the required English parser models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

### 4. Setup API Credentials
Create a `.env` file in the root directory and add your Anthropic and OpenAI API Keys:
```env
ANTHROPIC_API_KEY=your_claude_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Pipeline Execution Guide

Follow these steps sequentially to run the full pipeline.

### Step 1: Sentence Segmentation
Convert raw `.seg` text files into JSONL records with normal text and segments:
```bash
python data-preprocessing/segmentation.py raw-data/en/en.train.seg data-preprocessing/processed-data/data1.jsonl --id-prefix "EN_TR_"
```

### Step 2: Rewrite Segments using Claude
Complete fractured segments into standalone sentences. The output contains the `easy_read_rewritten` field:
```bash
python data-preprocessing/rewrite_easy_segments.py data-preprocessing/processed-data/data1.jsonl data-preprocessing/processed-data/rewritten_data.jsonl
```

### Step 3: Flag and Merge Incomplete Sentences (Post-Processing)
Identify sentences with issues (e.g. filler words, incorrect prepositions) for manual correction, and merge the corrections:
1. Run Step 1 flagging (uncomment Step 1 in `post-processing.py`):
   ```bash
   python data-preprocessing/post-processing.py
   ```
2. Manually correct the flagged items inside `segmentation_fixes_all.jsonl`.
3. Run Step 2 merge (uncomment Step 2 in `post-processing.py`):
   ```bash
   python data-preprocessing/post-processing.py
   ```

### Step 4: Parse AMR Graph Structure
You can generate AMR parses from the preprocessed data using the standalone Python CLI script:
```bash
python data-preprocessing/amr_generation.py data-preprocessing/processed-data/post_processed_segmentation_all.jsonl results/all_data.jsonl --batch-size 16
```
Alternatively, open and run `data-preprocessing/amr-generation.ipynb` in a Jupyter or Colab environment.

### Step 5: GNN Training, Inference, & Evaluation
You can run the full GNN training, structural guidance generation, LLM synthesis (OpenAI), and metrics evaluation using the standalone Python CLI script:
```bash
python training/gnn_training.py --all-data results/all_data.jsonl --epochs 30
```
*Note: If you do not have an `OPENAI_API_KEY` set and only want to train the GNN model locally on your machine, you can run with the `--skip-llm` flag:*
```bash
python training/gnn_training.py --all-data results/all_data.jsonl --epochs 30 --skip-llm
```
Alternatively, open and run `training/gnn-training.ipynb` in a Jupyter/Colab environment.

---

## License & Acknowledgements

- **License**: Distributed under the MIT License. See [LICENSE](LICENSE) for details.
- **Acknowledgements**: Raw datasets were provided courtesy of **Thierry Etchegoyhen**.