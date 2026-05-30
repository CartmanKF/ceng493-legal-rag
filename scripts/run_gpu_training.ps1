$ErrorActionPreference = "Stop"
$Python = ".\.conda-envs\legal-rag-gpu\python.exe"

& $Python scripts\prepare_full_finetune_data.py --dataset Datasets_Ceng493_legal_rag --out artifacts\full_finetune_data
& $Python scripts\gpu_train_embedding.py --dataset Datasets_Ceng493_legal_rag --max-steps 300 --out artifacts\gpu\embedding_model
& $Python scripts\gpu_train_reranker.py --dataset Datasets_Ceng493_legal_rag --max-steps 30 --batch-size 1 --grad-accum 4 --out artifacts\gpu\reranker_model
& $Python scripts\gpu_train_llm_lora.py --dataset Datasets_Ceng493_legal_rag --max-steps 60 --batch-size 1 --grad-accum 8 --max-length 384 --out artifacts\gpu\llm_lora
