import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .pipeline import build_pipeline


class LLMAnswerer:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.adapter_dir = root_dir / "artifacts" / "gpu" / "llm_lora"
        self.loaded_key = None
        self.tokenizer = None
        self.model = None

    def load(self, use_adapter: bool):
        key = "fine_tuned" if use_adapter else "base"
        if self.loaded_key == key and self.model is not None:
            return
        if self.model is not None:
            import gc
            import torch

            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization,
            device_map="auto",
            trust_remote_code=True,
        )
        if use_adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_dir)
        self.model.eval()
        self.loaded_key = key

    def answer(self, question: str, contexts: list[dict], use_adapter: bool) -> str:
        self.load(use_adapter)
        import torch

        blocks = []
        for index, context in enumerate(contexts[:3], start=1):
            blocks.append(
                "[{}] Baslik: {}\nCitation: {}\nMetin: {}".format(
                    index,
                    context.get("title") or "",
                    context.get("citation") or context.get("id"),
                    context.get("text") or "",
                )
            )
        messages = [
            {
                "role": "system",
                "content": "Sen bir Turk hukuku soru cevap asistanisin. Sadece verilen kaynaklari kullan. Cevap kisa, dogrudan ve Turkce olsun. Kaynak metnini tekrar yazma. Kaynakta yoksa bunu soyle.",
            },
            {"role": "user", "content": "Kaynaklar:\n\n" + "\n\n".join(blocks) + f"\n\nSoru: {question}\n\nCevap:"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=160,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        return self.clean_answer(answer)

    @staticmethod
    def clean_answer(answer: str) -> str:
        for marker in ["\n\nSoru:", "\nSoru:", "\n\nKaynaklar:", "\nKaynaklar:"]:
            if marker in answer:
                answer = answer.split(marker, 1)[0].strip()
        if "Cevap:" in answer:
            answer = answer.split("Cevap:", 1)[-1].strip()
        return answer


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
            use_adapter = self.mode_var.get() in {"fine_tuned", "adapted_llm", "full"}
            result["answer"] = self.llm_answerer.answer(question, result["contexts"], use_adapter)
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
