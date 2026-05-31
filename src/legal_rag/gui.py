import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .pipeline import build_pipeline


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
            self.pipeline = build_pipeline(self.dataset, self.artifacts, self.mode_var.get())
            self.result_queue.put(("status", "Hazir"))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def reload_pipeline(self):
        self.pipeline = None
        self.status_var.set("Model yukleniyor...")
        threading.Thread(target=self.load_pipeline, daemon=True).start()

    def change_mode(self):
        self.mode_var.set(self.mode_labels[self.mode_label_var.get()])
        self.reload_pipeline()

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
        self.status_var.set("Cevap hazirlaniyor...")
        self.answer.delete("1.0", tk.END)
        self.contexts.delete("1.0", tk.END)
        threading.Thread(target=self.answer_question, args=(text,), daemon=True).start()

    def answer_question(self, question: str):
        try:
            result = self.pipeline.answer(question)
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
        self.contexts.insert("1.0", "\n".join(lines))
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
