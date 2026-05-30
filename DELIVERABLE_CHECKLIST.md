# Deliverable Checklist

## Dataset

- [x] Used `Datasets_Ceng493_legal_rag`.
- [x] Did not use Kaggle `turkishlaw-dataset-for-llm-finetuning`.
- [x] Did not use Hugging Face `Renicames/turkish-lawchatbot`.
- [x] Gold benchmark has 240 verified Turkish legal QA rows.

## Code

- [x] Base RAG implementation.
- [x] Fine-tuned RAG implementation.
- [x] Same generator LLM family for base and fine-tuned systems.
- [x] Custom document input via `--custom-docs`.
- [x] Custom benchmark input via `--benchmark-file`.
- [x] GPU scripts for embedding, reranker, and LLM LoRA training.
- [x] GPU RAG evaluation script.

## Evaluation

- [x] Base vs fine-tuned comparison.
- [x] Ablation: adapted retriever.
- [x] Ablation: adapted reranker.
- [x] Ablation: adapted LLM behavior.
- [x] Retrieval metrics: Recall@k, MRR@10, nDCG.
- [x] QA metrics: EM, token F1, ROUGE-L.
- [x] Grounding metrics: citation hit, top-1 citation accuracy, grounded answer rate.
- [x] Hallucination proxies: faithfulness token support, unsupported sentence rate.
- [x] Custom data smoke test.

## Reports

- [x] Progress report Markdown.
- [x] Progress report PDF.
- [x] Final technical report Markdown.
- [x] Final technical report PDF.
- [x] Presentation deck or slide outline.
- [x] Local Git repository initialized.
- [ ] Remote GitHub link, if required by the submission system. This needs a GitHub remote URL from the team.

## Notes

- Large GPU model weight files are kept local under `artifacts/gpu` and ignored for normal Git commits because they exceed typical GitHub file limits.
- The code and training scripts are reproducible, so checkpoints can be recreated on GPU.
