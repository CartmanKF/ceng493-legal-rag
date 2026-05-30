---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-3B-Instruct
- lora
- transformers
---

# Turkish Legal RAG LoRA Adapter

This adapter was trained for the CENG493 Turkish Legal RAG project on the `Datasets_Ceng493_legal_rag` instruction data.

## Base Model

`Qwen/Qwen2.5-3B-Instruct`

## Training

| Field | Value |
|---|---:|
| Method | LoRA / QLoRA SFT |
| Steps | 60 |
| Average loss | 0.9744 |
| Trainable parameters | 14,966,784 |
| Hardware | NVIDIA GeForce RTX 4060 8GB |

## Intended Use

The adapter is used inside the project RAG pipeline to answer Turkish legal questions from retrieved source passages and include citations.

## Evaluation

The project reports retrieval, answer quality, citation, and faithfulness metrics in `reports/final_report.pdf` and `reports/evaluation_results_gold_full_metrics.json`.
