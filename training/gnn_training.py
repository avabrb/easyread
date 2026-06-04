#!/usr/bin/env python3
"""
GNN-Guided Text Simplification Training & Synthesis Pipeline.

Runs GNN training over semantic AMR document graphs, derives structural guidance,
builds LLM prompts, calls the OpenAI API to synthesize Easy Read texts, and computes
evaluation metrics (SARI, BLEU, Flesch-Kincaid, Concept Preservation).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import dotenv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import penman
import spacy
import syllables
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from easse.sari import corpus_sari
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import OpenAI
from scipy import stats as scipy_stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION ---


class EasyReadGNN(nn.Module):

    def __init__(self, in_channels=384, hidden_channels=128, dropout=0.4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, 64)
        self.fc_out = nn.Linear(64, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc_out(x).squeeze(-1)


# --- DATA LOADING & GRAPH BUILDING ---


def load_records(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def get_concept_embedding(concept_str: str, embedder: SentenceTransformer):
    base = concept_str.split("-")[0]
    return embedder.encode(base, convert_to_tensor=True)


def build_doc_graph(amr_strings: List[str]) -> nx.DiGraph:
    G = nx.DiGraph()
    sentence_roots = []
    concept_to_nodes = defaultdict(list)

    for sent_idx, amr_str in enumerate(amr_strings):
        try:
            g = penman.decode(amr_str)
        except Exception:
            continue

        instances = g.instances()
        if not instances:
            continue

        sent_root = None
        for inst in instances:
            node_id = f"{sent_idx}::{inst.source}"
            base_concept = inst.target.split("-")[0]
            G.add_node(
                node_id,
                concept=inst.target,
                base_concept=base_concept,
                sent_idx=sent_idx,
            )
            concept_to_nodes[base_concept].append(node_id)
            if sent_root is None:
                sent_root = node_id

        if sent_root:
            sentence_roots.append(sent_root)

        for edge in g.edges():
            src_id = f"{sent_idx}::{edge.source}"
            tgt_id = f"{sent_idx}::{edge.target}"
            if G.has_node(src_id) and G.has_node(tgt_id):
                G.add_edge(src_id, tgt_id, etype="semantic", role=edge.role)
                G.add_edge(
                    tgt_id, src_id, etype="semantic", role=edge.role + "_inv"
                )

    for i in range(len(sentence_roots) - 1):
        G.add_edge(
            sentence_roots[i],
            sentence_roots[i + 1],
            etype="discourse",
            role="NEXT_SENT",
        )

    for base_concept, node_ids in concept_to_nodes.items():
        if len(node_ids) > 1:
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    ni, nj = node_ids[i], node_ids[j]
                    si = G.nodes[ni].get("sent_idx", -1)
                    sj = G.nodes[nj].get("sent_idx", -1)
                    if si != sj:
                        G.add_edge(ni, nj, etype="coref", role="COREF")
                        G.add_edge(nj, ni, etype="coref", role="COREF")
    return G


def doc_graph_to_pyg(nx_graph: nx.DiGraph, label: int, embedder: SentenceTransformer) -> Optional[Data]:
    nodes = list(nx_graph.nodes())
    if not nodes:
        return None

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    x = torch.stack(
        [
            get_concept_embedding(
                nx_graph.nodes[n].get("concept", n.split("::")[-1]), embedder
            )
            for n in nodes
        ]
    )

    src_list, tgt_list = [], []
    for src, tgt in nx_graph.edges():
        if src in node_to_idx and tgt in node_to_idx:
            src_list.append(node_to_idx[src])
            tgt_list.append(node_to_idx[tgt])

    if src_list:
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


# --- TRAINING & VALIDATION LOOPS ---


def train_epoch(model, loader, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, loss_fn):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            total_loss += loss_fn(pred, batch.y).item()
            binary = (torch.sigmoid(pred) > 0.5).long().cpu().tolist()
            all_preds.extend(binary)
            all_labels.extend(batch.y.long().cpu().tolist())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f"Validation Loss: {avg_loss:.4f} | Acc: {acc:.3f} | F1: {f1:.3f}")
    return avg_loss


# --- INFERENCE & PROMPT BUILDING ---


def get_graph_embedding(model, data_obj: Data):
    model.eval()
    with torch.no_grad():
        data_obj = data_obj.to(device)
        x = F.relu(model.conv1(data_obj.x, data_obj.edge_index))
        x = F.relu(model.conv2(data_obj.x, data_obj.edge_index))
        embedding = global_mean_pool(
            x, torch.zeros(x.size(0), dtype=torch.long, device=device)
        )
        return embedding.cpu().numpy()


def embedding_to_structural_stats(nx_doc_graph: nx.DiGraph) -> dict:
    G = nx_doc_graph
    if G.number_of_nodes() == 0:
        return {
            "num_nodes": 0,
            "num_sentences": 1,
            "avg_nodes_per_sent": 0.0,
            "graph_density": 0.0,
            "max_path_length": 0,
            "num_coref_edges": 0,
            "top_concepts": [],
        }

    num_nodes = G.number_of_nodes()
    sent_indices = {
        data["sent_idx"]
        for _, data in G.nodes(data=True)
        if "sent_idx" in data
    }
    num_sentences = len(sent_indices) if sent_indices else 1
    avg_nodes_per_sent = round(num_nodes / num_sentences, 1)
    density = round(nx.density(G), 3)

    G_und = G.to_undirected()
    try:
        largest_cc = max(nx.connected_components(G_und), key=len)
        sub = G_und.subgraph(largest_cc)
        max_path = nx.diameter(sub)
    except Exception:
        max_path = 0

    num_coref = sum(
        1 for _, _, d in G.edges(data=True) if d.get("etype") == "coref"
    )

    degree_map = {
        G.nodes[n].get("base_concept", n): G.degree(n) for n in G.nodes()
    }
    top_concepts = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]

    return {
        "num_nodes": num_nodes,
        "num_sentences": num_sentences,
        "avg_nodes_per_sent": avg_nodes_per_sent,
        "graph_density": density,
        "max_path_length": max_path,
        "num_coref_edges": num_coref,
        "top_concepts": top_concepts,
    }


AMR_ABSTRACT_CONCEPTS = {
    "and",
    "or",
    "contrast",
    "have",
    "be",
    "do",
    "make",
    "get",
    "give",
    "say",
    "go",
    "know",
    "take",
    "see",
    "come",
    "want",
    "person",
    "thing",
    "place",
    "time",
    "way",
    "number",
    "name",
    "country",
    "organization",
    "before",
    "after",
    "cause",
    "multi-sentence",
    "relative-position",
    "they",
    "it",
    "she",
    "he",
    "we",
}


def derive_structural_guidance(stats: dict, easy_read_baseline=None) -> str:
    if easy_read_baseline is None:
        easy_read_baseline = {
            "avg_nodes_per_sent": 5.5,
            "max_path_length": 4,
            "graph_density": 0.35,
        }

    lines = ["STRUCTURAL GUIDANCE (from Easy Read graph analysis):"]

    current = stats["avg_nodes_per_sent"]
    target = easy_read_baseline["avg_nodes_per_sent"]
    if current > target * 1.3:
        lines.append(
            f"- This document averages {current} concepts per sentence "
            f"(Easy Read target: ~{target}). "
            f"Split long sentences so each expresses only one idea."
        )
    else:
        lines.append(
            f"- Sentence complexity is close to Easy Read level "
            f"({current} concepts/sentence). Maintain this simplicity."
        )

    depth = stats["max_path_length"]
    target_depth = easy_read_baseline["max_path_length"]
    if depth > target_depth:
        lines.append(
            f"- The concept chain depth is {depth} steps "
            f"(Easy Read target: <= {target_depth}). "
            f"Avoid nested clauses; flatten each idea to a direct statement."
        )
    else:
        lines.append(
            f"- Concept chain depth ({depth}) is within Easy Read range. "
            f"Keep sentences direct and avoid sub-clauses."
        )

    density = stats["graph_density"]
    if density < 0.15:
        lines.append(
            "- The document graph is sparse. Make logical connectives "
            "explicit in the output ('then', 'because', 'so', 'also')."
        )
    elif density > 0.5:
        lines.append(
            "- High concept overlap detected. Consolidate repeated ideas "
            "rather than restating them across sentences."
        )

    coref = stats["num_coref_edges"]
    if coref == 0:
        lines.append(
            "- No cross-sentence entity links detected. "
            "Use pronoun references or topic sentences to link ideas "
            "('He', 'This', 'They') so the output reads as connected text."
        )
    else:
        lines.append(
            f"- {coref} cross-sentence coreference links detected. "
            f"Preserve these references to maintain topic coherence."
        )

    num_nodes = stats["num_nodes"]
    target_sents = max(2, round(num_nodes / 5.5))
    lines.append(
        f"- Aim for around {target_sents} short sentences, but do not pad the output. "
        f"If the meaning is complete in fewer sentences, stop there. "
        f"Never add information that is not in the original text."
    )

    if stats["top_concepts"]:
        surface_concepts = [
            c
            for c, _ in stats["top_concepts"]
            if c.lower() not in AMR_ABSTRACT_CONCEPTS
        ]
        if surface_concepts:
            vocab = ", ".join(surface_concepts)
            lines.append(
                f"- Central concepts in this document: {vocab}. "
                f"Use these terms consistently in the output."
            )

    return "\n".join(lines)


def extract_knowledge_graph(text: str, nlp) -> tuple[nx.DiGraph, dict, list]:
    doc = nlp(text)
    G = nx.DiGraph()

    token_to_ent = {}
    for ent in doc.ents:
        G.add_node(ent.text, type=ent.label_, is_ent=True, is_stop=False)
        for token in ent:
            token_to_ent[token.i] = ent.text

    CONTENT_POS = {"NOUN", "PROPN", "VERB", "NUM", "ADJ"}
    for token in doc:
        if token.i in token_to_ent:
            continue
        node_id = token.text
        if node_id not in G:
            is_content = (
                token.pos_ in CONTENT_POS
                and not token.is_stop
                and not token.is_punct
            )
            G.add_node(
                node_id,
                type=token.pos_,
                is_ent=False,
                is_stop=(not is_content),
            )

    for token in doc:
        src_id = token_to_ent.get(token.i, token.text)
        tgt_id = token_to_ent.get(token.head.i, token.head.text)
        if src_id != tgt_id and G.has_node(src_id) and G.has_node(tgt_id):
            G.add_edge(src_id, tgt_id, relation=token.dep_)

    full_centrality = nx.degree_centrality(G)
    centrality = {
        node: score
        for node, score in full_centrality.items()
        if not G.nodes[node].get("is_stop", True)
        and not G.nodes[node].get("type", "") in {"PUNCT", "SPACE"}
    }

    ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
    return G, centrality, ranked


def build_llm_prompt(original_text: str, nx_doc_graph: nx.DiGraph, knowledge_graph_ranked: list) -> str:
    stats = embedding_to_structural_stats(nx_doc_graph)
    structural_guidance = derive_structural_guidance(stats)

    top_concepts = knowledge_graph_ranked[:10]
    if top_concepts:
        top_score = top_concepts[0][1]
        threshold = top_score * 0.5

        must = [(c, s) for c, s in top_concepts if s >= threshold]
        contextual = [(c, s) for c, s in top_concepts if s < threshold]

        must_lines = "\n".join(f"  {i+1}. {c}" for i, (c, _) in enumerate(must))
        concept_block = f"KEY CONCEPTS — you MUST use these words in your output:\n{must_lines}"

        if contextual:
            ctx_lines = "\n".join(
                f"  {i+1}. {c}" for i, (c, _) in enumerate(contextual)
            )
            concept_block += f"\n\nSUPPORTING CONCEPTS — include these only if they fit naturally:\n{ctx_lines}"
    else:
        concept_block = "KEY CONCEPTS: (none extracted)"

    prompt = f"""You are simplifying a document into Easy Read format for people with intellectual disabilities.

RULES:
- Use short sentences (8–12 words each).
- Use active voice.
- Express only one idea per sentence.
- Use simple, everyday words.
- Do not use jargon or technical terms without explaining them.
- Only use information from the original text. Do not add explanations,
  motivations, or details that are not explicitly stated.

{concept_block}

{structural_guidance}

ORIGINAL TEXT:
\"\"\"{original_text}\"\"\"

SIMPLIFIED OUTPUT:"""
    return prompt


def build_baseline_prompt(original_text: str) -> str:
    return f"""You are simplifying a document into Easy Read format for people with intellectual disabilities.

RULES:
- Use short sentences, about 8-12 words each.
- Use active voice.
- Express only one idea per sentence.
- Use simple, everyday words.
- Do not use jargon or technical terms without explaining them.
- Keep the important meaning of the original text.

ORIGINAL TEXT:
\"\"\"{original_text}\"\"\"

SIMPLIFIED OUTPUT:"""


def call_llm(client: OpenAI, prompt: str, model="gpt-4o-mini", max_retries=3) -> Optional[str]:
    if not client.api_key:
        raise RuntimeError("OpenAI API Client lacks an API Key. Check OPENAI_API_KEY environment variable.")
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2**attempt)
    return None


def get_reference(r: dict) -> Optional[str]:
    rewritten = r.get("easy_read_rewritten")
    if rewritten:
        return " ".join(rewritten) if isinstance(rewritten, list) else rewritten
    easy = r.get("easy_text")
    if easy:
        return " ".join(easy) if isinstance(easy, list) else easy
    return None


# --- METRICS & PLOTTING ---


def compute_sari(original, hypothesis, reference):
    return round(corpus_sari([original], [hypothesis], [[reference]]), 2)


def compute_bleu(hypothesis, reference):
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    sf = SmoothingFunction().method1
    return round(sentence_bleu([ref], hyp, smoothing_function=sf), 4)


def flesch_kincaid_grade(text):
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = [w for w in text.split() if w.strip()]
    if not sents or not words:
        return 0.0
    syl_count = sum(syllables.estimate(w) for w in words)
    asl = len(words) / len(sents)
    asw = syl_count / len(words)
    return round(0.39 * asl + 11.8 * asw - 15.59, 2)


def concept_preservation(output_text, top_concepts, top_n=5):
    out_lower = output_text.lower()
    must = top_concepts[:top_n]
    if not must:
        return 0.0
    found = sum(1 for concept, _ in must if concept.lower() in out_lower)
    return round(found / len(must), 3)


def avg_sentence_length(text):
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sents:
        return 0.0
    return round(sum(len(s.split()) for s in sents) / len(sents), 1)


def clean(text):
    return text.replace("  \n", " ").replace("\n", " ").strip()


def mean(vals):
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def generate_plots(scores_path: Path, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    with scores_path.open("r", encoding="utf-8") as f:
        scores = json.load(f)

    rows = scores["per_record"]
    corpus = scores["corpus"]

    ids = [str(r["id"]) for r in rows]
    x = np.arange(len(ids))
    width = 0.38

    # Plot 1: Corpus Metrics
    metrics = ["SARI", "BLEU", "Concept preservation", "Avg sentence length"]
    baseline = [
        corpus["sari"]["baseline"],
        corpus["bleu"]["baseline"],
        corpus["concept_preservation"]["baseline"],
        corpus["average_sentence_length"]["baseline"],
    ]
    graph = [
        corpus["sari"]["graph"],
        corpus["bleu"]["graph"],
        corpus["concept_preservation"]["graph"],
        corpus["average_sentence_length"]["graph"],
    ]
    x_metrics = np.arange(len(metrics))

    plt.figure(figsize=(10, 5))
    plt.bar(x_metrics - width / 2, baseline, width, label="Baseline")
    plt.bar(x_metrics + width / 2, graph, width, label="Graph-guided")
    plt.xticks(x_metrics, metrics, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Corpus-Level Evaluation: Baseline vs Graph-Guided")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "corpus_metric_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Per-Record SARI
    sari_base = [r["sari_base"] for r in rows]
    sari_graph = [r["sari_graph"] for r in rows]

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, sari_base, width, label="Baseline")
    plt.bar(x + width / 2, sari_graph, width, label="Graph-guided")
    plt.ylabel("SARI")
    plt.title("Per-Record SARI Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "per_record_sari.png", dpi=300)
    plt.close()

    # Plot 3: SARI Delta
    sari_delta = [r["sari_graph"] - r["sari_base"] for r in rows]
    plt.figure(figsize=(12, 5))
    plt.bar(ids, sari_delta)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Records")
    plt.xticks([])
    plt.ylabel("Graph SARI - Baseline SARI")
    plt.title("SARI Improvement from Graph Guidance")
    plt.tight_layout()
    plt.savefig(plots_dir / "sari_improvement.png", dpi=300)
    plt.close()

    # Plot 4: Readability (Flesch-Kincaid)
    fk_orig = corpus["flesch_kincaid_grade"]["original"]
    fk_base = corpus["flesch_kincaid_grade"]["baseline"]
    fk_graph = corpus["flesch_kincaid_grade"]["graph"]
    fk_ref = corpus["flesch_kincaid_grade"]["reference"]

    labels = ["Original", "Baseline", "Graph-guided", "Reference"]
    values = [fk_orig, fk_base, fk_graph, fk_ref]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Flesch-Kincaid Grade Level")
    plt.title("Readability Comparison")
    plt.tight_layout()
    plt.savefig(plots_dir / "fk_readability_comparison.png", dpi=300)
    plt.close()

    # Plot 5: Concept Preservation
    cp_base = [r["cp_base"] for r in rows]
    cp_graph = [r["cp_graph"] for r in rows]

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, cp_base, width, label="Baseline")
    plt.bar(x + width / 2, cp_graph, width, label="Graph-guided")
    plt.xlabel("Records")
    plt.xticks([])
    plt.ylabel("Concept Preservation Rate")
    plt.title("Per-Record Concept Preservation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "concept_preservation.png", dpi=300)
    plt.close()

    # Plot 6: BLEU distribution
    bleu_base = [r["bleu_base"] for r in rows]
    bleu_graph = [r["bleu_graph"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.hist(bleu_base, bins=40, alpha=0.6, label="Baseline", color="#1f77b4")
    plt.hist(bleu_graph, bins=40, alpha=0.6, label="Graph-guided", color="#ff7f0e")
    plt.axvline(
        corpus["bleu"]["baseline"],
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline mean ({corpus['bleu']['baseline']:.3f})",
    )
    plt.axvline(
        corpus["bleu"]["graph"],
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        label=f"Graph mean ({corpus['bleu']['graph']:.3f})",
    )
    plt.xlabel("BLEU Score")
    plt.ylabel("Number of Records")
    plt.title("BLEU Score Distribution: Baseline vs Graph-Guided")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "bleu_distribution.png", dpi=300)
    plt.close()


def fmt(val, decimals=3):
    return round(float(val), decimals)


def main():
    parser = argparse.ArgumentParser(
        description="GNN-Guided Text Simplification Pipeline"
    )
    parser.add_argument(
        "--all-data",
        type=Path,
        default=Path("results/all_data.jsonl"),
        help="Path to the input JSONL file containing texts and AMR parses.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("training/best_gnn.pt"),
        help="Path to save or load the GNN model weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train the GNN.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip GNN training and load the pretrained model.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM synthesis and evaluation steps.",
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=Path("results/llm_all_data_full_results.jsonl"),
        help="Output path to save full evaluation records.",
    )
    parser.add_argument(
        "--scores-json",
        type=Path,
        default=Path("results/evaluation_scores_all.json"),
        help="Output path to save scores JSON.",
    )
    parser.add_argument(
        "--paper-stats-json",
        type=Path,
        default=Path("results/paper_statistics.json"),
        help="Output path to save paper stats JSON.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results/plots_all"),
        help="Directory to save evaluation plots.",
    )
    args = parser.parse_args()

    if not args.skip_llm and not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please set it in a .env file or run with --skip-llm to only train the GNN."
        )

    # Step 1: Load records
    print(f"Loading AMR records from {args.all_data}...")
    records = load_records(args.all_data)
    print(f"Loaded {len(records)} records.")

    # Step 2: Initialize Sentence Transformer
    print("Loading Sentence Transformer embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Split records into Train & Validation
    train_records, val_records = train_test_split(
        records, test_size=0.2, random_state=42
    )

    model = EasyReadGNN(in_channels=384, hidden_channels=64).to(device)

    if not args.skip_training:
        print("Building document semantic graphs for GNN training dataset...")
        dataset = []
        for r in tqdm(train_records, desc="Processing Train Graphs"):
            normal_amrs = (
                r["normal_amr"]
                if isinstance(r["normal_amr"], list)
                else [r["normal_amr"]]
            )
            doc_G_normal = build_doc_graph(normal_amrs)
            pyg_normal = doc_graph_to_pyg(
                doc_G_normal, label=0, embedder=embedder
            )
            if pyg_normal is not None:
                dataset.append(pyg_normal)

            doc_G_easy = build_doc_graph(r["easy_amr"])
            pyg_easy = doc_graph_to_pyg(doc_G_easy, label=1, embedder=embedder)
            if pyg_easy is not None:
                dataset.append(pyg_easy)

        print(f"GNN Dataset size: {len(dataset)} graphs.")
        train_data, val_data = train_test_split(
            dataset,
            test_size=0.2,
            random_state=42,
            stratify=[d.y.item() for d in dataset],
        )
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=5e-4, weight_decay=1e-4
        )
        loss_fn = nn.BCEWithLogitsLoss()

        print("Starting GNN Model Training...")
        best_val_loss = float("inf")
        patience = 4
        bad_epochs = 0
        train_losses, val_losses = [], []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, epoch
            )
            val_loss = validate(model, val_loader, loss_fn)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                args.model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.model_path)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Loading best GNN weights from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path))
    else:
        print(f"Skipping training. Loading GNN weights from {args.model_path}...")
        if not args.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {args.model_path}. Train the model first."
            )
        model.load_state_dict(torch.load(args.model_path))

    if args.skip_llm:
        print("Skipping LLM synthesis and evaluation as requested by --skip-llm.")
        print("GNN Training Pipeline finished successfully.")
        return

    # Step 3: LLM Synthesis & Evaluation Loop
    print("Initializing LLM client & NLP resources...")
    nlp = spacy.load("en_core_web_sm")
    client = OpenAI(api_key=OPENAI_API_KEY)

    results = []
    print(f"Starting LLM evaluation on {len(val_records)} validation records...")

    for i, r in tqdm(
        list(enumerate(val_records)), desc="Evaluating & Synthesizing"
    ):
        original_text = r["normal_text"]
        reference = get_reference(r)

        if reference is None:
            continue

        # Baseline simplification
        baseline_prompt = build_baseline_prompt(original_text)
        baseline_output = call_llm(client, baseline_prompt)

        # Graph-guided simplification
        normal_amrs = r["normal_amr"] if isinstance(r["normal_amr"], list) else [r["normal_amr"]]
        doc_G = build_doc_graph(normal_amrs)
        _, _, ranked = extract_knowledge_graph(original_text, nlp)
        graph_prompt = build_llm_prompt(original_text, doc_G, ranked)
        graph_guided_output = call_llm(client, graph_prompt)

        if baseline_output is None or graph_guided_output is None:
            continue

        results.append(
            {
                "id": r.get("id", str(i)),
                "original_text": original_text,
                "reference_easy_text": reference,
                "baseline_output": baseline_output,
                "graph_guided_output": graph_guided_output,
                "top_concepts": ranked[:10],
                "graph_prompt": graph_prompt,
            }
        )
        time.sleep(1)

    print(f"Synthesized outputs for {len(results)} records.")

    # Write full results JSONL
    args.results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.results_jsonl.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Step 4: Metric Evaluation & Stats Generation
    rows = []
    for r in results:
        orig = r["original_text"]
        ref = r["reference_easy_text"]
        base = clean(r["baseline_output"])
        graph = clean(r["graph_guided_output"])
        concepts = r["top_concepts"]

        rows.append(
            {
                "id": r["id"],
                "sari_base": compute_sari(orig, base, ref),
                "sari_graph": compute_sari(orig, graph, ref),
                "bleu_base": compute_bleu(base, ref),
                "bleu_graph": compute_bleu(graph, ref),
                "fk_orig": flesch_kincaid_grade(orig),
                "fk_ref": flesch_kincaid_grade(ref),
                "fk_base": flesch_kincaid_grade(base),
                "fk_graph": flesch_kincaid_grade(graph),
                "cp_base": concept_preservation(base, concepts),
                "cp_graph": concept_preservation(graph, concepts),
                "asl_base": avg_sentence_length(base),
                "asl_graph": avg_sentence_length(graph),
            }
        )

    sari_b = mean([r["sari_base"] for r in rows])
    sari_g = mean([r["sari_graph"] for r in rows])
    bleu_b = mean([r["bleu_base"] for r in rows])
    bleu_g = mean([r["bleu_graph"] for r in rows])
    fk_orig = mean([r["fk_orig"] for r in rows])
    fk_b = mean([r["fk_base"] for r in rows])
    fk_g = mean([r["fk_graph"] for r in rows])
    fk_ref = mean([r["fk_ref"] for r in rows])
    cp_b = mean([r["cp_base"] for r in rows])
    cp_g = mean([r["cp_graph"] for r in rows])
    asl_b = mean([r["asl_base"] for r in rows])
    asl_g = mean([r["asl_graph"] for r in rows])
    sari_delta = round(sari_g - sari_b, 3)

    # Save scores JSON
    scores_output = {
        "n_records": len(rows),
        "corpus": {
            "sari": {"baseline": sari_b, "graph": sari_g, "delta": sari_delta},
            "bleu": {"baseline": bleu_b, "graph": bleu_g},
            "flesch_kincaid_grade": {
                "original": fk_orig,
                "baseline": fk_b,
                "graph": fk_g,
                "reference": fk_ref,
            },
            "concept_preservation": {"baseline": cp_b, "graph": cp_g},
            "average_sentence_length": {"baseline": asl_b, "graph": asl_g},
        },
        "per_record": rows,
    }

    args.scores_json.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_json.open("w", encoding="utf-8") as f:
        json.dump(scores_output, f, indent=2, ensure_ascii=False)

    # Paired t-tests
    sari_b_arr = np.array([r["sari_base"] for r in rows])
    sari_g_arr = np.array([r["sari_graph"] for r in rows])
    bleu_b_arr = np.array([r["bleu_base"] for r in rows])
    bleu_g_arr = np.array([r["bleu_graph"] for r in rows])
    fk_orig_arr = np.array([r["fk_orig"] for r in rows])
    fk_ref_arr = np.array([r["fk_ref"] for r in rows])
    fk_b_arr = np.array([r["fk_base"] for r in rows])
    fk_g_arr = np.array([r["fk_graph"] for r in rows])
    cp_b_arr = np.array([r["cp_base"] for r in rows])
    cp_g_arr = np.array([r["cp_graph"] for r in rows])
    asl_b_arr = np.array([r["asl_base"] for r in rows])
    asl_g_arr = np.array([r["asl_graph"] for r in rows])

    sari_delta_arr = sari_g_arr - sari_b_arr
    bleu_delta_arr = bleu_g_arr - bleu_b_arr
    cp_delta_arr = cp_g_arr - cp_b_arr
    fk_delta_arr = fk_g_arr - fk_b_arr

    wins = int(np.sum(sari_delta_arr > 0))
    losses = int(np.sum(sari_delta_arr < 0))
    ties = int(np.sum(sari_delta_arr == 0))
    N_total = len(rows)

    t_sari, p_sari = scipy_stats.ttest_rel(sari_g_arr, sari_b_arr)
    t_bleu, p_bleu = scipy_stats.ttest_rel(bleu_g_arr, bleu_b_arr)
    t_fk, p_fk = scipy_stats.ttest_rel(fk_g_arr, fk_b_arr)
    t_cp, p_cp = scipy_stats.ttest_rel(cp_g_arr, cp_b_arr)

    paper_stats = {
        "n_records": N_total,
        "corpus_means": {
            "sari": {
                "baseline": fmt(np.mean(sari_b_arr)),
                "graph": fmt(np.mean(sari_g_arr)),
                "delta": fmt(np.mean(sari_delta_arr)),
            },
            "bleu": {
                "baseline": fmt(np.mean(bleu_b_arr)),
                "graph": fmt(np.mean(bleu_g_arr)),
                "delta": fmt(np.mean(bleu_delta_arr)),
            },
            "fk": {
                "original": fmt(np.mean(fk_orig_arr)),
                "reference": fmt(np.mean(fk_ref_arr)),
                "baseline": fmt(np.mean(fk_b_arr)),
                "graph": fmt(np.mean(fk_g_arr)),
                "delta": fmt(np.mean(fk_delta_arr)),
            },
            "cp": {
                "baseline": fmt(np.mean(cp_b_arr)),
                "graph": fmt(np.mean(cp_g_arr)),
                "delta": fmt(np.mean(cp_delta_arr)),
            },
            "asl": {
                "baseline": fmt(np.mean(asl_b_arr)),
                "graph": fmt(np.mean(asl_g_arr)),
            },
        },
        "standard_deviations": {
            "sari": {
                "baseline": fmt(np.std(sari_b_arr)),
                "graph": fmt(np.std(sari_g_arr)),
            },
            "bleu": {
                "baseline": fmt(np.std(bleu_b_arr)),
                "graph": fmt(np.std(bleu_g_arr)),
            },
            "fk": {
                "baseline": fmt(np.std(fk_b_arr)),
                "graph": fmt(np.std(fk_g_arr)),
            },
            "cp": {
                "baseline": fmt(np.std(cp_b_arr)),
                "graph": fmt(np.std(cp_g_arr)),
            },
            "asl": {
                "baseline": fmt(np.std(asl_b_arr)),
                "graph": fmt(np.std(asl_g_arr)),
            },
        },
        "win_loss_tie": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": fmt(wins / N_total),
            "win_pct": fmt(wins / N_total * 100, 1),
            "loss_pct": fmt(losses / N_total * 100, 1),
            "tie_pct": fmt(ties / N_total * 100, 1),
        },
        "sari_delta": {
            "mean": fmt(np.mean(sari_delta_arr)),
            "sd": fmt(np.std(sari_delta_arr)),
            "median": fmt(np.median(sari_delta_arr)),
            "min": fmt(np.min(sari_delta_arr)),
            "max": fmt(np.max(sari_delta_arr)),
            "n_gt2": int(np.sum(sari_delta_arr > 2)),
            "n_lt_neg2": int(np.sum(sari_delta_arr < -2)),
        },
        "readability": {
            "pct_baseline_below_original": fmt(
                np.mean(fk_b_arr < fk_orig_arr) * 100, 1
            ),
            "pct_graph_below_original": fmt(
                np.mean(fk_g_arr < fk_orig_arr) * 100, 1
            ),
            "graph_vs_reference_delta": fmt(
                np.mean(fk_g_arr) - np.mean(fk_ref_arr)
            ),
        },
        "significance_tests": {
            "sari": {"t": fmt(t_sari, 4), "p": fmt(p_sari, 4)},
            "bleu": {"t": fmt(t_bleu, 4), "p": fmt(p_bleu, 4)},
            "fk": {"t": fmt(t_fk, 4), "p": fmt(p_fk, 4)},
            "cp": {"t": fmt(t_cp, 4), "p": fmt(p_cp, 4)},
        },
        "concept_preservation_detail": {
            "pct_graph_perfect": fmt(np.mean(cp_g_arr == 1.0) * 100, 1),
            "pct_baseline_perfect": fmt(np.mean(cp_b_arr == 1.0) * 100, 1),
            "pct_graph_beats_baseline": fmt(
                np.mean(cp_g_arr > cp_b_arr) * 100, 1
            ),
            "pct_graph_worse": fmt(np.mean(cp_g_arr < cp_b_arr) * 100, 1),
        },
    }

    args.paper_stats_json.parent.mkdir(parents=True, exist_ok=True)
    with args.paper_stats_json.open("w", encoding="utf-8") as f:
        json.dump(paper_stats, f, indent=2, ensure_ascii=False)

    print("Generating and saving evaluation plots...")
    generate_plots(args.scores_json, args.plots_dir)

    print("GNN Training & LLM Synthesis Pipeline finished successfully.")


if __name__ == "__main__":
    main()
