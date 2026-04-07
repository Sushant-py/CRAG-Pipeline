import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from logic_engine import run_logic_engine
from langchain_groq import ChatGroq

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_FILE = "golden_dataset.json"
TOTAL_DOCS   = 60

# ── Load grader models once at startup ─────────────────────────────────────────

print("Loading grader models...")

# Grader 1 — LLM via Groq 
# Complex, general-purpose reasoning.
llm_grader = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Grader 2 — Embedding Cosine Similarity 
# Simplest baseline. Encodes fact and ground truth independently, then compares.
embed_model      = SentenceTransformer('BAAI/bge-small-en-v1.5')
COSINE_THRESHOLD = 0.50

print("All grader models loaded.\n")


# ── Grader 1: LLM ─────────────────────────────────────────────────────────────
def grade_with_llm(query: str, ground_truth: str, fact: str,
                   strict: bool = False) -> bool:
    """
    Uses LLM to judge relevance with natural language reasoning.
    strict=True  → combined Tier-1 check (does the whole set capture ground truth?)
    strict=False → individual Tier-2 check (is this single chunk relevant?)
    """
    if strict:
        task = ("Does the 'Retrieved Context' contain the specific information "
                "found in the 'Ground Truth Info'?")
    else:
        task = ("Does the 'Retrieved Context' contain information relevant and "
                "useful for answering the Query, even if it only covers part of "
                "the answer?")

    prompt = f"""
Query: {query}
Ground Truth Info: {ground_truth}
Retrieved Context: {fact}

Task: {task}
Respond with ONLY 'YES' or 'NO'.
"""
    try:
        response = llm_grader.invoke(prompt).content.strip().upper()
        return "YES" in response
    except Exception as e:
        print(f"  [!] LLM Grader Error: {e}")
        return False


# ── Grader 2: Embedding Cosine Similarity ─────────────────────────────────────
def grade_with_cosine(query: str, ground_truth: str, fact: str) -> bool:
    """
    Compares BGE embedding of retrieved fact against BGE embedding of ground truth.
    """
    vec_gt   = embed_model.encode([ground_truth])
    vec_fact = embed_model.encode([fact])
    sim      = float(cosine_similarity(vec_gt, vec_fact)[0][0])
    return sim >= COSINE_THRESHOLD


# ── Metric calculation ─────────────────────────────────────────────────────────
def compute_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    accuracy  = ((tp + tn) / (tp + tn + fp + fn)
                 if (tp + tn + fp + fn) > 0 else 0)
    return precision, recall, f1, accuracy


# ── Per-grader plots (Confusion Matrix + Metrics Bar) ─────────────────────────
def plot_for_grader(grader_name, tp, fp, fn, tn,
                    precision, recall, f1, accuracy):
    """
    One confusion matrix + one metrics bar chart per grader.
    """
    safe = grader_name.replace(" ", "_").replace("(", "").replace(")", "").replace("—","").replace("-","").lower()
    safe = "_".join(safe.split())

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Negative (Ignored)", "Positive (Kept)"],
        yticklabels=["Negative (Irrelevant)", "Positive (Relevant)"],
    )
    plt.title(f"CRAG — {grader_name}\nConfusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    fname_cm = f"confusion_matrix_{safe}.png"
    plt.savefig(fname_cm, dpi=300)
    plt.close()
    print(f"  └─ Saved {fname_cm}")

    # Metrics bar chart
    plt.figure(figsize=(8, 5))
    metrics_labels = ["Precision", "Recall", "F1 Score", "Accuracy"]
    scores         = [precision, recall, f1, accuracy]
    ax             = sns.barplot(x=metrics_labels, y=scores, palette="viridis")
    plt.ylim(0, 1.15)
    plt.title(f"CRAG — {grader_name}\nPerformance Metrics")
    plt.ylabel("Score (0.0 to 1.0)")
    for i, v in enumerate(scores):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    fname_mc = f"metrics_chart_{safe}.png"
    plt.savefig(fname_mc, dpi=300)
    plt.close()
    print(f"  └─ Saved {fname_mc}")


# ── Grouped comparison chart ──────────────────────────────────────────────────
def plot_comparison_chart(all_results: dict):
    """
    Grouped bar chart comparing the graders across all 4 metrics.
    """
    metric_keys  = ["precision", "recall", "f1", "accuracy"]
    metric_labels = ["Precision", "Recall", "F1 Score", "Accuracy"]
    grader_names = list(all_results.keys())
    
    # Updated to 2 colors since we only have 2 graders now
    colors       = ["#1D9E75", "#D85A30"]

    x     = np.arange(len(metric_labels))
    width = 0.35  # Widened the bars slightly since there are fewer

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (grader, metrics) in enumerate(all_results.items()):
        vals = [metrics[k] for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width,
                      label=grader, color=colors[i % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}", ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title(
        "CRAG Grader Comparison\nLLM vs Cosine Similarity",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("grader_comparison_chart.png", dpi=300)
    plt.close()
    print("  └─ Saved grader_comparison_chart.png")


# ── Core evaluation loop ───────────────────────────────────────────────────────
def evaluate_with_grader(test_cases, grader_fn, grader_name):
    tp, fp, fn, tn = 0, 0, 0, 0
    total = len(test_cases)

    print(f"\n{'━'*70}")
    print(f"  GRADER: {grader_name}")
    print(f"{'━'*70}")

    for i, case in enumerate(test_cases):
        query    = case["question"]
        gt       = case["contexts"][0]

        print(f"\n  Test {i+1}/{total}: '{query[:55]}...'")

        pipeline    = run_logic_engine(query)
        final_facts = pipeline["facts"]

        print(f"  Pipeline returned {len(final_facts)} fact(s).")

        # Empty pipeline → definite miss
        if not final_facts:
            fn += 1
            tn += TOTAL_DOCS - 1
            print(f"  └─ Empty → FN")
            print("-" * 50)
            continue

        # ── Tier 1: combined LLM check ─────────────────────────────────────────
        combined_ok = grade_with_llm(
            query, gt, "\n".join(final_facts), strict=True
        )

        if not combined_ok:
            fn += 1
            fp += len(final_facts)
            tn += max(0, (TOTAL_DOCS - 1) - len(final_facts))
            print(f"  └─ Combined check FAILED → FN + {len(final_facts)} FPs")
            print("-" * 50)
            continue

        # ── Tier 2: per-chunk grading with THIS grader ────────────────────────
        print(f"  └─ Combined check PASSED. Grading individually with {grader_name}...")
        for j, fact in enumerate(final_facts):
            relevant = grader_fn(query, gt, fact)
            if relevant:
                tp += 1
                print(f"     Fact {j+1} → TP")
            else:
                fp += 1
                print(f"     Fact {j+1} → FP")

        tn += max(0, (TOTAL_DOCS - 1) - len(final_facts))
        print("-" * 50)

    precision, recall, f1, accuracy = compute_metrics(tp, fp, fn, tn)

    print(f"\n  Raw counts → TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={precision:.4f}  Recall={recall:.4f}  "
          f"F1={f1:.4f}  Accuracy={accuracy:.4f}")

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def run_ml_evaluation():
    if not os.path.exists(DATASET_FILE):
        print(f"Error: {DATASET_FILE} not found!")
        return

    with open(DATASET_FILE, "r") as f:
        all_cases = json.load(f)

    test_cases = all_cases[:10]

    # ── Removed CrossEncoder, keeping only LLM and Cosine ─────────────────────
    graders = [
        (grade_with_llm,          "LLM (llama-3.1-8b)"),
        (grade_with_cosine,       "Embedding Cosine Similarity"),
    ]

    all_results = {}
    for grader_fn, grader_name in graders:
        result = evaluate_with_grader(test_cases, grader_fn, grader_name)
        all_results[grader_name] = result

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  GRADER COMPARISON TABLE")
    print("━" * 70)
    print(f"  {'Grader':<40} {'Accuracy':>9} {'Precision':>9} "
          f"{'Recall':>9} {'F1':>9}")
    print("  " + "-" * 68)
    for name, m in all_results.items():
        print(f"  {name:<40} {m['accuracy']:>9.4f} {m['precision']:>9.4f} "
              f"{m['recall']:>9.4f} {m['f1']:>9.4f}")
    print("━" * 70)

    best_name = max(all_results, key=lambda n: all_results[n]["f1"])
    print(f"\n  Best grader by F1: {best_name}")
    print("━" * 70)

    # ── Generate all visualizations ───────────────────────────────────────────
    print("\nGenerating visualizations...")
    for name, m in all_results.items():
        plot_for_grader(
            name,
            m["tp"], m["fp"], m["fn"], m["tn"],
            m["precision"], m["recall"], m["f1"], m["accuracy"],
        )

    plot_comparison_chart(all_results)
    print("\nAll visualizations saved.")


if __name__ == "__main__":
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run_ml_evaluation()