import unittest

from src.legal_rag.pipeline import RAGPipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_returns_citation(self):
        docs = [
            {
                "id": "doc1",
                "title": "Ceza Hukuku",
                "text": "Madde 225 hüküm iddianamede gösterilen fiil ve fail hakkında verilir.",
                "metadata": {"citation_label": "Test Citation - doc1"},
            }
        ]
        result = RAGPipeline(docs, fine_tuned_style=True).answer("Madde 225 hüküm ne hakkındadır?")
        self.assertIn("doc1", result["contexts"][0]["id"])
        self.assertIn("Kaynaklar", result["answer"])


if __name__ == "__main__":
    unittest.main()
