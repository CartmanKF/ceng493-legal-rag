# Turkish Legal RAG - CENG493

Turkish legal question answering project with Base RAG, Fine-tuned RAG, ablation evaluation, custom document support, and custom benchmark support.

The default dataset directory is `Datasets_Ceng493_legal_rag`.

## Project Contents

- `src/legal_rag`: RAG pipeline, retrieval, reranking, answer generation, metrics, GUI.
- `scripts/evaluate.py`: Base/Fine-tuned/Ablation benchmark evaluation.
- `scripts/gpu_rag_evaluate.py`: GPU neural RAG evaluation with dense retrieval and LLM generation.
- `scripts/gpu_train_*.py`: GPU fine-tuning scripts.
- `artifacts`: lightweight adapter weights and GPU model artifacts.
- `examples`: small custom document and benchmark examples.
- `reports/final_report.pdf`: final report.
- `RAG_Arayuz.bat`: local desktop interface launcher.

## Install

Clone the repository:

```powershell
git clone https://github.com/CartmanKF/ceng493-legal-rag.git
cd ceng493-legal-rag
```

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

The custom benchmark evaluation script uses the lightweight RAG pipeline and can run on CPU with the Python standard library.

Install the full package list only if the GPU training scripts or the desktop Qwen/LoRA generation path will be used:

```powershell
pip install -r requirements.txt
```

GPU is needed for the neural Qwen/LoRA scripts and the desktop LLM generation path.

## Run the Default Benchmark

This command evaluates Base RAG, Fine-tuned RAG, and ablation modes on the included gold benchmark:

```powershell
python scripts\evaluate.py --dataset Datasets_Ceng493_legal_rag --artifacts artifacts --benchmark gold --out reports\evaluation_results_gold_full_metrics.json
```

It evaluates these modes on the same benchmark:

- `base`
- `adapted_retriever`
- `adapted_reranker`
- `adapted_llm`
- `fine_tuned`

Main metrics:

- Recall@1, Recall@3, Recall@5, Recall@10
- MRR@10
- nDCG@5, nDCG@10
- Exact Match
- Token F1
- ROUGE-L
- Citation hit rate
- Top-1 citation accuracy
- Faithfulness token support
- Unsupported sentence rate

## Test With Custom Documents and Custom Benchmark

The evaluator can provide both a custom document collection and a custom benchmark file.

```powershell
python scripts\evaluate.py --dataset Datasets_Ceng493_legal_rag --artifacts artifacts --custom-docs path\to\docs.jsonl --benchmark-file path\to\benchmark.json --out reports\teacher_eval.json
```

The same command works with a directory of `.txt` files:

```powershell
python scripts\evaluate.py --dataset Datasets_Ceng493_legal_rag --artifacts artifacts --custom-docs path\to\doc_folder --benchmark-file path\to\benchmark.json --out reports\teacher_eval.json
```

### Custom Document Formats

One `.txt` file:

```text
Legal document text...
```

Folder of `.txt` files:

```text
custom_docs/
  doc1.txt
  doc2.txt
```

JSONL:

```json
{"id": "doc1", "title": "Example Article", "text": "Document text.", "metadata": {"citation_label": "Example m.1"}}
```

JSON list:

```json
[
  {
    "id": "doc1",
    "title": "Example Article",
    "text": "Document text.",
    "metadata": {
      "citation_label": "Example m.1"
    }
  }
]
```

### Custom Benchmark Format

```json
[
  {
    "question": "Question text",
    "verified_answer": "Gold answer text",
    "gold_sources": [
      {
        "source_id": "doc1",
        "corpus_row_id": "doc1",
        "citation_label": "Example m.1"
      }
    ]
  }
]
```

If relevant document ids are available, put them in `gold_sources`. Retrieval and citation metrics use those ids.

## Included Custom Data Example

Run:

```powershell
python scripts\evaluate.py --dataset Datasets_Ceng493_legal_rag --artifacts artifacts --custom-docs examples\custom_docs.jsonl --benchmark-file examples\custom_benchmark.json --out reports\custom_docs_benchmark_smoke.json
```

Expected smoke-test behavior:

- Recall@1 = 1.0
- Citation hit rate = 1.0
- Top-1 citation accuracy = 1.0

## Ask One Question From the Command Line

```powershell
python -m src.legal_rag.cli ask --dataset Datasets_Ceng493_legal_rag --artifacts artifacts --mode fine_tuned --question "Ceza Muhakemesi Kanunu m.225 nasil duzenlenmistir?"
```

Available modes:

- `base`
- `adapted_retriever`
- `adapted_reranker`
- `adapted_llm`
- `fine_tuned`

## Desktop Interface

On the prepared local environment, run:

```powershell
.\RAG_Arayuz.bat
```

The desktop interface retrieves sources from the RAG pipeline and generates an answer with `Qwen/Qwen2.5-3B-Instruct`. Fine-tuned answer mode uses the LoRA adapter under `artifacts\gpu\llm_lora`.

The first answer can take longer because the LLM is loaded into memory.

The same interface can also be used with a custom document collection:

- `Dokuman dosyasi sec`: select a `.jsonl`, `.json`, or `.txt` document file.
- `Dokuman klasoru sec`: select a folder that contains `.txt` documents.
- `Varsayilana don`: switch back to `Datasets_Ceng493_legal_rag`.

Custom benchmark evaluation can be started from the interface:

- `Benchmark sec`: select a benchmark JSON file.
- `Benchmark calistir`: compare Base RAG, Fine-tuned RAG, and the ablation modes on the selected benchmark.

The benchmark JSON format is the same format shown in the custom benchmark section above.

## GPU Training Commands

```powershell
python scripts\gpu_train_embedding.py --dataset Datasets_Ceng493_legal_rag --max-steps 300 --out artifacts\gpu\embedding_model
python scripts\gpu_train_reranker.py --dataset Datasets_Ceng493_legal_rag --max-steps 30 --batch-size 1 --grad-accum 4 --out artifacts\gpu\reranker_model
python scripts\gpu_train_llm_lora.py --dataset Datasets_Ceng493_legal_rag --max-steps 60 --batch-size 1 --grad-accum 8 --max-length 384 --out artifacts\gpu\llm_lora
```

Training summaries:

- Embedding: 300 steps, average loss 0.0561.
- Reranker: 30 steps, average loss 0.3735.
- LLM LoRA: 60 steps, average loss 0.9744.

## GPU Neural Evaluation

```powershell
python scripts\gpu_rag_evaluate.py --dataset Datasets_Ceng493_legal_rag --benchmark gold --limit 20 --no-generation --out reports\gpu_rag_retrieval_gold_20.json
python scripts\gpu_rag_evaluate.py --dataset Datasets_Ceng493_legal_rag --benchmark gold --limit 1 --out reports\gpu_rag_generation_gold_1.json
```

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Main Results

Full 240-question gold benchmark:

| System | R@1 | R@10 | MRR@10 | nDCG@10 | F1 | ROUGE-L | Citation hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.875 | 0.979 | 0.913 | 0.929 | 0.395 | 0.370 | 0.954 |
| Fine-tuned | 0.879 | 0.979 | 0.914 | 0.930 | 0.398 | 0.373 | 0.946 |

The detailed discussion is in `reports/final_report.pdf`.
