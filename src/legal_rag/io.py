import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_corpus(dataset_dir: Path) -> list[dict]:
    return read_jsonl(dataset_dir / "corpus.jsonl")


def load_custom_documents(path: Path) -> list[dict]:
    if path.is_dir():
        docs = []
        for file_path in sorted(path.glob("*.txt")):
            docs.append(
                {
                    "id": file_path.stem,
                    "title": file_path.name,
                    "text": file_path.read_text(encoding="utf-8"),
                    "metadata": {"source_file": str(file_path)},
                }
            )
        return docs
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".json":
        data = read_json(path)
        if isinstance(data, list):
            return data
        raise ValueError("Custom JSON document collection must be a list of objects.")
    if path.suffix.lower() == ".txt":
        return [{"id": path.stem, "title": path.name, "text": path.read_text(encoding="utf-8"), "metadata": {}}]
    raise ValueError(f"Unsupported custom document format: {path}")


def batched(items: list, size: int) -> Iterable[list]:
    for index in range(0, len(items), size):
        yield items[index : index + size]

