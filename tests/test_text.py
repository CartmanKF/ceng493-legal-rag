import unittest

from src.legal_rag.text import token_f1, tokenize


class TextTests(unittest.TestCase):
    def test_tokenize_turkish_text(self):
        self.assertIn("susma", tokenize("Susma hakkı kişinin özgürce karar vermesidir."))

    def test_token_f1_overlap(self):
        self.assertGreater(token_f1("Madde 225 hüküm iddianame", "Madde 225 hüküm"), 0.5)


if __name__ == "__main__":
    unittest.main()
