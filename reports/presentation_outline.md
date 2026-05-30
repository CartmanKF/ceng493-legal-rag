# 15-Minute Presentation Outline

## Slide 1 - Title

Turkish Legal RAG: Base vs Fine-tuned RAG Comparison

Speaker note: Introduce the project goal: reliable Turkish legal question answering with source-grounded RAG.

## Slide 2 - Assignment Requirements

- Compare Base RAG and Fine-tuned RAG with the same LLM.
- Use a gold QA benchmark.
- Evaluate retrieval, answer quality, citations, and hallucination risk.
- Support evaluator-provided custom documents and custom benchmark questions.
- Show ablation results for retriever, reranker, and LLM components.

Speaker note: Emphasize that the implementation targets the grading rubric directly.

## Slide 3 - Dataset Decision

- Used only `Datasets_Ceng493_legal_rag`.
- Did not use Kaggle `turkishlaw-dataset-for-llm-finetuning`.
- Did not use Hugging Face `Renicames/turkish-lawchatbot`.
- Gold benchmark: 240 verified Turkish legal QA rows.

Speaker note: This answers the dataset substitution clearly.

## Slide 4 - System Architecture

Question -> Retriever -> Candidate documents -> Reranker -> Context -> Same LLM -> Answer with citations

Speaker note: Explain that Base and Fine-tuned systems differ by adapted components, not by switching to another LLM family.

## Slide 5 - Model Choices

| Component | Model |
|---|---|
| LLM | `Qwen/Qwen2.5-3B-Instruct` |
| Embedding | `intfloat/multilingual-e5-base` |
| Reranker | `BAAI/bge-reranker-v2-m3` |

Speaker note: Qwen was selected for Turkish instruction capability and 8GB GPU feasibility with LoRA/QLoRA.

## Slide 6 - Fine-tuning Strategy

- Embedding: triplet contrastive training.
- Reranker: cross-encoder binary relevance training.
- LLM: LoRA/QLoRA supervised fine-tuning.
- Prompting: source-only legal answer with citations.

Speaker note: Mention all training used GPU.

## Slide 7 - GPU Training Summary

| Component | Steps | Avg loss |
|---|---:|---:|
| Embedding | 300 | 0.0561 |
| Reranker | 30 | 0.3735 |
| LLM LoRA | 60 | 0.9744 |

Speaker note: Checkpoints are under `artifacts/gpu`.

## Slide 8 - Evaluation Metrics

- Retrieval: Recall@1/3/5/10, MRR@10, nDCG@5/10.
- Answer: Exact Match, token F1, ROUGE-L.
- Grounding: citation hit, top-1 citation accuracy.
- Hallucination proxy: faithfulness token support, unsupported sentence rate.

Speaker note: Explain why Exact Match is too strict for citation-based legal answers.

## Slide 9 - Full Gold Benchmark Results

| System | R@1 | R@10 | MRR@10 | F1 | Citation hit |
|---|---:|---:|---:|---:|---:|
| Base | 0.875 | 0.979 | 0.913 | 0.395 | 0.954 |
| Fine-tuned | 0.879 | 0.979 | 0.914 | 0.398 | 0.946 |

Speaker note: Present this as a modest improvement, not an exaggerated one.

## Slide 10 - GPU Neural Smoke Result

| System | R@1 | F1 | Citation hit | Faithfulness |
|---|---:|---:|---:|---:|
| Base neural RAG | 0.00 | 0.250 | 0.00 | 0.375 |
| Fine-tuned neural RAG | 1.00 | 0.870 | 1.00 | 0.690 |

Speaker note: This validates that the actual GPU LoRA path works end-to-end.

## Slide 11 - Ablation Study

- Base.
- Adapted retriever.
- Adapted reranker.
- Adapted LLM behavior.
- Fully fine-tuned RAG.

Speaker note: The ablation is implemented and repeatable through `scripts/evaluate.py`.

## Slide 12 - Custom Evaluation Support

Command:

```powershell
.\.conda-envs\legal-rag-gpu\python.exe scripts\evaluate.py --custom-docs path\to\docs.jsonl --benchmark-file path\to\benchmark.json --artifacts artifacts --out reports\teacher_eval.json
```

Speaker note: This directly satisfies the requirement that the teacher can provide custom documents and questions.

## Slide 13 - Error Analysis

- Base neural RAG can answer from a nearby but wrong legal article.
- Embedding-only fine-tuning is mixed and not always better.
- Generated answers are rarely exact string matches, so F1/ROUGE/citation metrics are more meaningful.
- Unsupported sentence rate helps flag hallucination risk.

Speaker note: Be honest about limitations; this usually helps evaluation credibility.

## Slide 14 - Demo Plan

- Run one `ask` command on the default dataset.
- Run `scripts/evaluate.py` on `examples/custom_docs.jsonl` and `examples/custom_benchmark.json`.
- Show report JSON and citations.

Speaker note: The custom data smoke test already passes with Recall@1 = 1.0.

## Slide 15 - Conclusion

- Project uses the requested dataset.
- Base vs Fine-tuned RAG comparison is implemented.
- Gold and custom benchmarks are supported.
- Ablation, GPU training, and hallucination-oriented metrics are included.
- Remaining work: longer neural training and cached dense embeddings for larger GPU evaluation.
