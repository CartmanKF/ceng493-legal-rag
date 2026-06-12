import json
import queue
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tempfile import TemporaryDirectory
from tkinter import filedialog, messagebox, ttk

from .evaluation import evaluate_rag
from .io import read_json
from .pipeline import build_pipeline

EVALUATION_MODES = ["base", "adapted_retriever", "adapted_reranker", "adapted_llm", "fine_tuned"]


class LLMAnswerer:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.adapter_dir = root_dir / "artifacts" / "gpu" / "llm_lora"

    def answer(self, question: str, contexts: list[dict], draft_answer: str, use_adapter: bool) -> str:
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "input.json"
            output_path = temp / "output.json"
            input_path.write_text(
                json.dumps({"question": question, "contexts": contexts, "draft_answer": draft_answer}, ensure_ascii=False),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(self.root_dir / "scripts" / "gui_llm_generate.py"),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--model",
                self.model_name,
            ]
            if use_adapter:
                command.extend(["--adapter", str(self.adapter_dir)])
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            startupinfo = None
            creationflags = 0
            if sys.platform.startswith("win"):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                creationflags = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run(
                command,
                cwd=self.root_dir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "LLM generation failed.")
            return json.loads(output_path.read_text(encoding="utf-8"))["answer"]


def project_root() -> Path:
    candidates = [
        Path.cwd(),
        Path(sys.executable).resolve().parent,
        Path(sys.executable).resolve().parent.parent,
        Path(__file__).resolve().parents[2],
    ]
    for candidate in candidates:
        if (candidate / "Datasets_Ceng493_legal_rag").exists() and (candidate / "artifacts").exists():
            return candidate
    return Path.cwd()


def run_benchmark(dataset: Path, artifacts: Path, custom_docs: Path | None, benchmark_path: Path) -> dict:
    benchmark = read_json(benchmark_path)
    if not isinstance(benchmark, list):
        raise ValueError("Benchmark file must contain a JSON list.")
    results = {}
    for mode in EVALUATION_MODES:
        pipeline = build_pipeline(dataset, artifacts, mode, custom_docs)
        results[mode] = evaluate_rag(pipeline, benchmark)
    return results


class LegalRAGApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Turkish Legal RAG")
        self.root.geometry("980x720")
        self.root.minsize(760, 560)

        self.root_dir = project_root()
        self.dataset = self.root_dir / "Datasets_Ceng493_legal_rag"
        self.artifacts = self.root_dir / "artifacts"
        self.pipeline = None
        self.custom_docs_path = None
        self.custom_benchmark_path = None
        self.llm_answerer = LLMAnswerer(self.root_dir)
        self.result_queue = queue.Queue()

        self.mode_var = tk.StringVar(value="fine_tuned")
        self.status_var = tk.StringVar(value="Model yukleniyor...")
        self.mode_labels = {
            "Base RAG": "base",
            "Fine-tuned RAG": "fine_tuned",
            "Sadece retriever fine-tuned": "adapted_retriever",
            "Sadece reranker fine-tuned": "adapted_reranker",
            "Sadece cevap stili fine-tuned": "adapted_llm",
        }
        self.label_by_mode = {value: key for key, value in self.mode_labels.items()}
        self.mode_label_var = tk.StringVar(value=self.label_by_mode[self.mode_var.get()])
        self.docs_var = tk.StringVar(value="Varsayilan dokuman koleksiyonu")
        self.benchmark_var = tk.StringVar(value="Benchmark secilmedi")

        self.build_ui()
        threading.Thread(target=self.load_pipeline, daemon=True).start()
        self.root.after(100, self.poll_result_queue)

    def build_ui(self):
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text="Turkish Legal RAG", font=("Segoe UI", 18, "bold"))
        title.pack(anchor=tk.W)

        top = ttk.Frame(container)
        top.pack(fill=tk.X, pady=(14, 8))

        ttk.Label(top, text="Mod").pack(side=tk.LEFT)
        mode = ttk.Combobox(
            top,
            textvariable=self.mode_label_var,
            values=list(self.mode_labels.keys()),
            state="readonly",
            width=30,
        )
        mode.pack(side=tk.LEFT, padx=(8, 16))
        mode.bind("<<ComboboxSelected>>", lambda _: self.change_mode())

        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT)

        ttk.Label(container, text="Soru").pack(anchor=tk.W, pady=(8, 4))
        question_frame = ttk.Frame(container)
        question_frame.pack(fill=tk.X)

        self.question = tk.Text(question_frame, height=4, wrap=tk.WORD, font=("Segoe UI", 11))
        self.question.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.question.bind("<Control-Return>", lambda _: self.ask())

        ask_button = ttk.Button(question_frame, text="Sor", command=self.ask)
        ask_button.pack(side=tk.LEFT, padx=(10, 0), fill=tk.Y)

        samples = ttk.Frame(container)
        samples.pack(fill=tk.X, pady=(10, 8))
        sample_questions = [
            "Ceza Muhakemesi Kanunu m.225 nasil duzenlenmistir?",
            "Turk Medeni Kanunu m.307 tek basina evlat edinme hakkinda ne soyler?",
            "Bilgi Edinme Hakki Kanunu m.17 neyi duzenler?",
        ]
        for sample in sample_questions:
            ttk.Button(samples, text=sample, command=lambda text=sample: self.set_question(text)).pack(side=tk.LEFT, padx=(0, 8))

        data_tools = ttk.Labelframe(container, text="Custom veri ve benchmark", padding=8)
        data_tools.pack(fill=tk.X, pady=(2, 8))

        data_row = ttk.Frame(data_tools)
        data_row.pack(fill=tk.X)
        ttk.Button(data_row, text="Dokuman dosyasi sec", command=self.select_custom_docs_file).pack(side=tk.LEFT)
        ttk.Button(data_row, text="Dokuman klasoru sec", command=self.select_custom_docs_folder).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(data_row, text="Varsayilana don", command=self.clear_custom_docs).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(data_row, textvariable=self.docs_var).pack(side=tk.LEFT, padx=(12, 0))

        benchmark_row = ttk.Frame(data_tools)
        benchmark_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(benchmark_row, text="Benchmark sec", command=self.select_benchmark).pack(side=tk.LEFT)
        ttk.Button(benchmark_row, text="Benchmark calistir", command=self.evaluate_benchmark).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(benchmark_row, textvariable=self.benchmark_var).pack(side=tk.LEFT, padx=(12, 0))

        panes = ttk.PanedWindow(container, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        answer_frame = ttk.Labelframe(panes, text="Cevap", padding=8)
        self.answer = tk.Text(answer_frame, wrap=tk.WORD, font=("Segoe UI", 11))
        self.answer.pack(fill=tk.BOTH, expand=True)
        panes.add(answer_frame, weight=3)

        context_frame = ttk.Labelframe(panes, text="Kaynaklar", padding=8)
        self.contexts = tk.Text(context_frame, wrap=tk.WORD, font=("Consolas", 10), height=8)
        self.contexts.pack(fill=tk.BOTH, expand=True)
        panes.add(context_frame, weight=1)

        bottom = ttk.Frame(container)
        bottom.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(bottom, text="JSON kopyala", command=self.copy_json).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Temizle", command=self.clear).pack(side=tk.LEFT, padx=(8, 0))

        self.last_result = None

    def load_pipeline(self):
        try:
            self.pipeline = build_pipeline(self.dataset, self.artifacts, self.mode_var.get(), self.custom_docs_path)
            self.result_queue.put(("status", "Hazir - LLM cevap modu"))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def reload_pipeline(self):
        self.pipeline = None
        self.status_var.set("Model yukleniyor...")
        threading.Thread(target=self.load_pipeline, daemon=True).start()

    def change_mode(self):
        self.mode_var.set(self.mode_labels[self.mode_label_var.get()])
        self.reload_pipeline()

    def select_custom_docs_file(self):
        selected = filedialog.askopenfilename(
            title="Custom dokuman sec",
            filetypes=[
                ("Supported files", "*.jsonl *.json *.txt"),
                ("JSONL", "*.jsonl"),
                ("JSON", "*.json"),
                ("Text", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.custom_docs_path = Path(selected)
            self.docs_var.set(str(self.custom_docs_path))
            self.reload_pipeline()

    def select_custom_docs_folder(self):
        selected = filedialog.askdirectory(title="Custom dokuman klasoru sec")
        if selected:
            self.custom_docs_path = Path(selected)
            self.docs_var.set(str(self.custom_docs_path))
            self.reload_pipeline()

    def clear_custom_docs(self):
        self.custom_docs_path = None
        self.docs_var.set("Varsayilan dokuman koleksiyonu")
        self.reload_pipeline()

    def select_benchmark(self):
        selected = filedialog.askopenfilename(
            title="Benchmark JSON sec",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.custom_benchmark_path = Path(selected)
            self.benchmark_var.set(str(self.custom_benchmark_path))

    def evaluate_benchmark(self):
        benchmark_path = self.custom_benchmark_path or (self.dataset / "gold_benchmark.json")
        if not benchmark_path.exists():
            messagebox.showinfo("Benchmark gerekli", "Benchmark JSON dosyasi sec.")
            return
        self.status_var.set("Benchmark calisiyor...")
        self.answer.delete("1.0", tk.END)
        self.contexts.delete("1.0", tk.END)
        threading.Thread(target=self.evaluate_benchmark_worker, args=(benchmark_path,), daemon=True).start()

    def evaluate_benchmark_worker(self, benchmark_path: Path):
        try:
            results = run_benchmark(self.dataset, self.artifacts, self.custom_docs_path, benchmark_path)
            self.result_queue.put(("benchmark", results))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def set_question(self, text: str):
        self.question.delete("1.0", tk.END)
        self.question.insert("1.0", text)

    def ask(self):
        text = self.question.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Soru gerekli", "Once bir soru yaz.")
            return
        if self.pipeline is None:
            messagebox.showinfo("Hazir degil", "Pipeline henuz yukleniyor.")
            return
        self.status_var.set("LLM cevap uretiyor...")
        self.answer.delete("1.0", tk.END)
        self.contexts.delete("1.0", tk.END)
        threading.Thread(target=self.answer_question, args=(text,), daemon=True).start()

    def answer_question(self, question: str):
        try:
            result = self.pipeline.answer(question)
            draft_answer = result["answer"]
            use_adapter = self.mode_var.get() in {"fine_tuned", "adapted_llm", "full"}
            result["answer"] = self.llm_answerer.answer(question, result["contexts"], draft_answer, use_adapter)
            result["generator"] = "Qwen/Qwen2.5-3B-Instruct" + (" + LoRA" if use_adapter else "")
            self.result_queue.put(("answer", result))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def poll_result_queue(self):
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "answer":
                    self.show_answer(payload)
                elif kind == "benchmark":
                    self.show_benchmark(payload)
                elif kind == "error":
                    self.status_var.set("Hata")
                    messagebox.showerror("Hata", payload)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_result_queue)

    def show_answer(self, result: dict):
        self.last_result = result
        self.answer.delete("1.0", tk.END)
        self.answer.insert("1.0", self.answer_without_sources(result["answer"]))
        self.contexts.delete("1.0", tk.END)
        lines = []
        for index, context in enumerate(result["contexts"], start=1):
            lines.append(f"{index}. {context['id']}")
            lines.append(f"   {context['citation']}")
        if result.get("generator"):
            lines.append("")
            lines.append(f"LLM: {result['generator']}")
        self.contexts.insert("1.0", "\n".join(lines))
        self.status_var.set("Hazir")

    def show_benchmark(self, results: dict):
        lines = ["Mode\tN\tR@1\tR@10\tMRR@10\tF1\tCitation\tFaithfulness"]
        for mode, metrics in results.items():
            lines.append(
                "\t".join(
                    [
                        mode,
                        str(metrics["n"]),
                        f"{metrics['recall@1']:.3f}",
                        f"{metrics['recall@10']:.3f}",
                        f"{metrics['mrr@10']:.3f}",
                        f"{metrics['answer_token_f1']:.3f}",
                        f"{metrics['citation_hit_rate']:.3f}",
                        f"{metrics['faithfulness_token_support']:.3f}",
                    ]
                )
            )
        self.answer.delete("1.0", tk.END)
        self.answer.insert("1.0", "\n".join(lines))
        self.contexts.delete("1.0", tk.END)
        self.contexts.insert("1.0", "Benchmark sonucu arayuzde gosterildi. Ayrintili JSON icin komut satiri evaluate.py kullanilabilir.")
        self.last_result = {"benchmark": results}
        self.status_var.set("Hazir")

    @staticmethod
    def answer_without_sources(answer: str) -> str:
        for marker in ["\n\nKaynaklar:", "\n\nCitations:"]:
            if marker in answer:
                return answer.split(marker, 1)[0].strip()
        return answer.strip()

    def copy_json(self):
        if not self.last_result:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(json.dumps(self.last_result, ensure_ascii=False, indent=2))
        self.status_var.set("JSON panoya kopyalandi")

    def clear(self):
        self.question.delete("1.0", tk.END)
        self.answer.delete("1.0", tk.END)
        self.contexts.delete("1.0", tk.END)
        self.last_result = None


def main():
    root = tk.Tk()
    app = LegalRAGApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
