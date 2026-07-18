# 🚀 BEIR Benchmark Evaluation of a Fine-Tuned BERT Dense Retriever

This repository presents the **zero-shot evaluation** of my custom **BERT Dense Retriever**, fine-tuned from **bert-base-uncased** on the **MS MARCO Passage Ranking** dataset.

The model is evaluated on multiple datasets from the **BEIR (Benchmarking Information Retrieval)** benchmark to measure its ability to generalize across diverse retrieval tasks without additional fine-tuning. The evaluation also compares the proposed retrieval pipeline against the **original BM25 lexical baseline** reported by BEIR.

> **Model:** https://huggingface.co/Innovatewithapple/bert-dense-retriever

---

# 📖 About BEIR

**BEIR (Benchmarking Information Retrieval)** is one of the most widely used zero-shot evaluation benchmarks for information retrieval. It consists of heterogeneous datasets spanning multiple domains, enabling researchers to evaluate how well retrieval models generalize beyond their training data.

This repository evaluates the model on the following BEIR datasets:

| Dataset | Domain |
|----------|--------|
| FEVER | Fact Verification |
| Quora | Duplicate Question Retrieval |
| HotpotQA | Multi-Hop Question Answering |
| FiQA | Financial Retrieval |
| TREC-COVID | Biomedical Literature Retrieval |

---

# 🏗️ Evaluation Pipeline

The retrieval pipeline follows a two-stage architecture:

```
User Query
     │
     ▼
Dense Retriever
(Custom BERT Dense Retriever)
     │
 Top-K Candidate Retrieval
     │
     ▼
Cross Encoder
(ms-marco-MiniLM-L-6-v2)
     │
     ▼
Final Ranked Results
```

For specialized domains such as biomedical retrieval (TREC-COVID), a **Hybrid Retrieval** strategy combining **BM25** and the dense retriever is used before Cross-Encoder re-ranking to improve candidate recall.

---

# 📊 BEIR Evaluation Results

The table below compares the proposed retrieval pipeline against the **BM25 baseline reported by the original BEIR benchmark**.

| Dataset | Retrieval Strategy | BM25 (BEIR) NDCG@10 | Pipeline NDCG@10 | Recall@10 | Recall@100 | Improvement |
|----------|-------------------|--------------------:|-----------------:|-----------:|------------:|------------:|
| FEVER | Hybrid + Cross Encoder | 0.7530 | **0.9791** | **0.9873** | **0.9937** | **+0.2261** |
| Quora | Dense + Cross Encoder | 0.7830 | **0.9686** | **0.9858** | **0.9938** | **+0.1856** |
| HotpotQA | Dense + Cross Encoder | 0.6030 | **0.8977** | **0.8853** | **0.8960** | **+0.2947** |
| FiQA | Dense + Cross Encoder | 0.2361 | **0.7512** | **0.8055** | — | **+0.5151** |
| TREC-COVID | Hybrid + Cross Encoder | 0.6559 | **0.6868** | **0.0181** | **0.1110** | **+0.0309** |

> **Note:** BM25 scores are taken from the original BEIR benchmark and are included as the lexical retrieval baseline for comparison.

---

# 🔍 Key Findings

- Strong zero-shot generalization across multiple retrieval domains.
- Significant improvements over the BEIR BM25 baseline on FEVER, Quora, HotpotQA, and FiQA.
- Hybrid retrieval (BM25 + Dense Retriever) improves retrieval quality for specialized biomedical documents in TREC-COVID.
- Cross-Encoder re-ranking substantially enhances the final ranking quality by leveraging full query-document interactions.

---

# 📂 Repository Structure

```
bert-dense-retriever-benchmark/
│
├── EvaluateBeirBenchMark_FEVERDataset.ipynb
├── EvaluateBeirBenchMark_HOTPOTQADataset.ipynb
├── EvaluateBeirBenchMark_QuoraDataset.ipynb
├── EvaluateBeirBenchMark_fiqaDataset.ipynb
├── EvaluateBeirBenchMark_TrecCovidDataset.ipynb
└── README.md
```

Each notebook contains the complete evaluation pipeline for its respective BEIR dataset, including retrieval, optional hybrid search, Cross-Encoder re-ranking, and metric computation.

---

# 🤖 Evaluated Model

The evaluation uses the following custom dense retrieval model:

**Innovatewithapple/bert-dense-retriever**

- Base Model: `bert-base-uncased`
- Fine-tuned on: MS MARCO Passage Ranking
- Embedding Type: Dense Retrieval
- Framework: Hugging Face Transformers

Model Repository:
https://huggingface.co/Innovatewithapple/bert-dense-retriever

---

# 📚 References

**BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**

```
@inproceedings{thakur2021beir,
  title={BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  author={Thakur, Nandan and others},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2021}
}
```

---

# ⭐ Acknowledgements

This evaluation builds upon the BEIR benchmarking framework while assessing the zero-shot performance of a custom fine-tuned BERT Dense Retriever across multiple retrieval domains.
