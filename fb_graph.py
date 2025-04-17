#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Dynamic Feedback Knowledge Graph

Features:
- If fewer than 4 texts: each text forms its own cluster
- Otherwise, determine optimal number of clusters in [K_MIN, min(unique_embeddings, K_MAX_LIMIT)] by silhouette score
- Use Openrouter (Deepseek) to extract one core keyword per cluster
- Project cluster centers to 1D and map to a “candy” Pastel1 gradient: closer = more similar colors (but all distinct)
- Node size ∝ number of texts in cluster
- Every pair of clusters has an edge whose thickness ∝ cosine similarity
"""

import os
import random
import re
import requests
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

# —— Configuration Parameters —— #
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

EMBED_MODEL    = "all-MiniLM-L6-v2"
SIM_THRESHOLD  = 0.4   # used to scale edge widths
ITERATIONS     = 20
K_MIN          = 2
K_MAX_LIMIT    = 10

# Edge width settings
MIN_EDGE_WIDTH = 1
MAX_EDGE_WIDTH = 6

# Matplotlib settings for Chinese fonts (retain if needed)
plt.rcParams["font.sans-serif"]    = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# Suggestion / opinion pools
suggestion_pool = [
    "Add a modular plugin marketplace for easier extensions",
    "Support exporting to PDF and Excel formats",
    "Include confirmation dialogs before critical operations",
    "Allow customizable keyboard shortcuts",
    "Improve mobile adaptation for better touch experience",
    "Add customizable filters for visual reports",
    "Show error stack traces in logs",
    "Provide interactive API documentation and sandbox",
    "Enable real-time collaborative editing",
    "Support batch import and export of data",
    "Offer dark mode and high-contrast themes",
    "Include a user behavior analytics dashboard",
    "Implement fine-grained permission controls",
    "Add automatic data backup and restore features",
    "Allow markdown in email notifications"
]
opinion_pool = [
    "The system occasionally crashes",
    "CAPTCHA fails to load on login",
    "Upload progress bar stalls after file upload",
    "Search results do not match keywords",
    "UI buttons sometimes unresponsive",
    "Exported files have formatting issues",
    "Table scroll performance is laggy",
    "User avatar upload fails",
    "Chart loading speed is too slow",
    "Encoding issues after language switch",
    "No friendly prompts in registration flow",
    "Floating menus obscure content on mobile",
    "Account deletion does not work",
    "No confirmation message after feedback submission",
    "Push notifications delayed over a minute"
]

suggestion_feedback = []
opinion_feedback    = []

model = SentenceTransformer(EMBED_MODEL)


def extract_keyword_deepseek(text: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a keyword extraction assistant that returns one keyword."},
            {"role": "user",   "content": f"Extract the most representative keyword from the following text:\n{text}"}
        ],
        "temperature": 0.0
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip().split()[0]
    except Exception:
        parts = re.split(r'[,，;\s]+', text)
        return parts[0] if parts else "—"


def auto_cluster_and_keyword(texts):
    n = len(texts)
    if n == 0:
        return {}, {}, {}
    emb = model.encode(texts)

    # fewer than 4 texts => singleton clusters
    if n < 4:
        clusters = {i: [texts[i]] for i in range(n)}
        centers  = {i: emb[i]      for i in range(n)}
        keywords = {i: extract_keyword_deepseek(texts[i]) for i in range(n)}
        return clusters, centers, keywords

    uniq = np.unique(emb, axis=0).shape[0]
    if uniq < 2:
        clusters = {0: texts}
        centers  = {0: emb.mean(axis=0)}
        keywords = {0: extract_keyword_deepseek(" ".join(texts))}
        return clusters, centers, keywords

    best_score   = -1
    best_labels  = None
    best_centers = None
    for k in range(K_MIN, min(uniq, K_MAX_LIMIT) + 1):
        km     = KMeans(n_clusters=k, random_state=42).fit(emb)
        labels = km.labels_
        if 2 <= np.unique(labels).shape[0] <= n - 1:
            score = silhouette_score(emb, labels)
        else:
            score = -1
        if score > best_score:
            best_score   = score
            best_labels  = labels
            best_centers = km.cluster_centers_

    clusters = {}
    for lbl, txt in zip(best_labels, texts):
        clusters.setdefault(lbl, []).append(txt)
    centers  = {lbl: best_centers[lbl] for lbl in clusters}
    keywords = {lbl: extract_keyword_deepseek(" ".join(clusters[lbl])) for lbl in clusters}
    return clusters, centers, keywords


def build_graph(clusters, centers, keywords):
    G = nx.Graph()
    for lbl, segs in clusters.items():
        count = len(segs)
        G.add_node(lbl,
                   size  = count * 300,
                   label = f"{keywords[lbl]}\n{count} items")
    keys = list(centers.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            sim  = cosine_similarity([centers[a]], [centers[b]])[0][0]
            G.add_edge(a, b, weight=sim)
    return G


def plot_graph(G, centers, ax, title):
    ax.cla()
    if G.number_of_nodes() == 0:
        ax.set_title(f"{title}\n(No data)")
        ax.axis("off")
        return

    # 1) Compute layout
    pos = nx.spring_layout(G, seed=42)

    # 2) Sort nodes by x-coordinate for unique but smoothly varying "candy" colors
    nodes_sorted = sorted(G.nodes(), key=lambda n: pos[n][0])
    cmap = plt.cm.Pastel1
    colors_map = {}
    total = len(nodes_sorted)
    for idx, n in enumerate(nodes_sorted):
        t = idx / (total - 1) if total > 1 else 0.5
        colors_map[n] = cmap(t)
    node_colors = [colors_map[n] for n in G.nodes()]

    # 3) Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size  = [G.nodes[n]["size"] for n in G.nodes()],
        node_color = node_colors,
        alpha      = 0.9,
        ax         = ax
    )

    # 4) Draw labels
    nx.draw_networkx_labels(
        G, pos,
        labels    = nx.get_node_attributes(G, "label"),
        font_size = 10,
        ax        = ax
    )

    # 5) Draw edges with thickness ∝ similarity
    edge_list = []
    widths    = []
    for u, v, d in G.edges(data=True):
        sim = d["weight"]
        # scale thickness
        w   = MIN_EDGE_WIDTH + (sim - SIM_THRESHOLD) / (1 - SIM_THRESHOLD) * (MAX_EDGE_WIDTH - MIN_EDGE_WIDTH)
        edge_list.append((u, v))
        widths.append(w)
    nx.draw_networkx_edges(
        G, pos,
        edgelist   = edge_list,
        width      = widths,
        edge_color = "gray",
        alpha      = 0.5,
        ax         = ax
    )

    ax.set_title(title, fontweight="bold")
    ax.axis("off")


def main():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i in range(1, ITERATIONS + 1):
        suggestion_feedback.append(random.choice(suggestion_pool))
        opinion_feedback.append(random.choice(opinion_pool))

        cs, centers_s, kws_s = auto_cluster_and_keyword(suggestion_feedback)
        co, centers_o, kws_o = auto_cluster_and_keyword(opinion_feedback)

        Gs = build_graph(cs, centers_s, kws_s)
        Go = build_graph(co, centers_o, kws_o)

        plot_graph(Gs, centers_s, ax1, "Suggestion Feedback Graph")
        plot_graph(Go, centers_o, ax2, "Issue Feedback Graph")

        print(f"Iteration {i}/{ITERATIONS} — Suggestions: {len(suggestion_feedback)}, Issues: {len(opinion_feedback)}")
        plt.pause(1)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
