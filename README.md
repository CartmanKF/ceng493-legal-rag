# Turkish Legal RAG - CENG493

This repository contains a reproducible Turkish legal RAG project using the provided `Datasets_Ceng493_legal_rag` dataset instead of the datasets listed in the original assignment PDF.

The Kaggle `turkishlaw-dataset-for-llm-finetuning` and Hugging Face `Renicames/turkish-lawchatbot` datasets were not used.

## What Is Included

- Base RAG and Fine-tuned RAG comparison with the same generator LLM family.
- Custom document collection support.
- Custom benchmark support.
- Gold benchmark evaluation support.
- Ablation study for adapted retriever, adapted reranker, and adapted answer behavior.
- GPU fine-tuning scripts for embedding, reranker, and LLM LoRA/QLoRA.
- GPU neural RAG evaluation script.
- Progress report PDF, final technical report PDF, deliverable checklist, and presentation outline.

## Dataset

Expected default dataset directory:

```text
Datasets_Ceng493_legal_rag/
  corpus.jsonl
  embedding.jsonl
  reranker.jsonl
  llm.jsonl
  gold_benchmark.json
  rag_eval.json
```

Current dataset sizes:

- Corpus: 7,579 rows.
- Embedding training: 2,059 rows.
- Reranker training: 6,752 rows.
- LLM training: 13,758 rows.
- Gold benchmark: 240 verified QA rows.
- RAG eval set: 1,000 rows.

## Quick Start

Launch the desktop interface:

```powershell
.\RAG_Arayuz.exe
```

If Windows blocks the executable, use the batch launcher:

```powershell
.\RAG_Arayuz.bat
```

Train lightweight domain adapters:

```powershell
python scripts/train_lightweight.py --dataset Datasets_Ceng493_legal_rag --out artifacts
```

Run the full gold benchmark with all implemented metrics:

```powershell
.\.conda-envs\legal-rag-gpu\python.exe scripts\evaluate.py --dataset Datasets_Ceng493_legal_rag --artifacts artifacts --benchmark gold --out reports\evaluation_results_gold_full_metrics.json
```

Ask a question:

```powershell
python -m src.legal_rag.cli ask --dataset Datasets_Ceng493_legal_rag --artifacts artifacts --mode fine_tuned --question "Ceza Muhakemesi Kanunu m.225 nasil duzenlenmistir?"
```

## Custom Documents and Benchmarks

Custom documents may be:

- A directory of `.txt` files.
- A single `.txt` file.
- `.jsonl` rows with `id`, `text`, optional `title`, and optional `metadata.citation_label`.
- A `.json` list of the same objects.

Run evaluation on evaluator-provided documents and evaluator-provided benchmark:

```powershell
.\.conda-envs\legal-rag-gpu\python.exe scripts\evaluate.py --custom-docs path\to\docs.jsonl --benchmark-file path\to\benchmark.json --artifacts artifacts --out reports\teacher_eval.json
```

Smoke-test examples:

- `examples/custom_docs.jsonl`
- `examples/custom_benchmark.json`
- `reports/custom_docs_benchmark_smoke.json`

## GPU Training

The GPU environment was validated with CUDA on NVIDIA GeForce RTX 4060 8GB.

```powershell
.\.conda-envs\legal-rag-gpu\python.exe scripts\gpu_train_embedding.py --dataset Datasets_Ceng493_legal_rag --max-steps 300 --out artifacts\gpu\embedding_model
.\.conda-envs\legal-rag-gpu\python.exe scripts\gpu_train_reranker.py --dataset Datasets_Ceng493_legal_rag --max-steps 30 --batch-size 1 --grad-accum 4 --out artifacts\gpu\reranker_model
.\.conda-envs\legal-rag-gpu\python.exe scripts\gpu_train_llm_lora.py --dataset Datasets_Ceng493_legal_rag --max-steps 60 --batch-size 1 --grad-accum 8 --max-length 384 --out artifacts\gpu\llm_lora
```

Training summaries:

- Embedding: 300 steps, average loss 0.0561.
- Reranker: 30 steps, average loss 0.3735.
- LLM LoRA: 60 steps, average loss 0.9744.

## GPU Neural RAG Evaluation

```powershell
.\.conda-envs\legal-rag-gpu\python.exe scripts\gpu_rag_evaluate.py --dataset Datasets_Ceng493_legal_rag --benchmark gold --limit 20 --no-generation --out reports\gpu_rag_retrieval_gold_20.json
.\.conda-envs\legal-rag-gpu\python.exe scripts\gpu_rag_evaluate.py --dataset Datasets_Ceng493_legal_rag --benchmark gold --limit 1 --out reports\gpu_rag_generation_gold_1.json
```

## Metrics

The project reports:

- Retrieval: Recall@1, Recall@3, Recall@5, Recall@10, MRR@10, nDCG@5, nDCG@10.
- QA: Exact Match, token F1, ROUGE-L.
- Grounding: citation hit rate, top-1 citation accuracy, grounded answer rate.
- Hallucination proxies: faithfulness token support and unsupported sentence rate.

## Current Key Results

Full 240-question lightweight gold benchmark:

| System | R@1 | R@10 | MRR@10 | nDCG@10 | F1 | ROUGE-L | Citation hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.875 | 0.979 | 0.913 | 0.929 | 0.395 | 0.370 | 0.954 |
| Fine-tuned | 0.879 | 0.979 | 0.914 | 0.930 | 0.398 | 0.373 | 0.946 |

GPU neural smoke tests:

- Retrieval-only, first 20 gold rows: Base Recall@1 `0.60`, Fine-tuned Recall@1 `0.85`.
- Generation, first gold row: Base F1 `0.250`, Fine-tuned F1 `0.870`.

## Reports and Deliverables

- `reports/progress_report.pdf`
- `reports/final_report.pdf`
- `reports/presentation_outline.md`
- `DELIVERABLE_CHECKLIST.md`

## Tests

```powershell
.\.conda-envs\legal-rag-gpu\python.exe -m unittest discover -s tests -p "test_*.py"
```

The latest run passed 3 tests.
