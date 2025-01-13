from typing import Dict, List

from PyPDF2 import PdfReader

from aibrary import AiBrary


class SimpleRAGSystem:
    def __init__(
        self,
        aibrary: AiBrary,
        embeddings: dict = {},
        model_name="gpt-4",
        embedding_model="text-embedding-3-small",
    ):
        self.aibrary = aibrary
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.embeddings = embeddings

    def load_pdf(self, pdf_path: str) -> List[str]:
        """Loads and extracts text from a PDF file."""
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() for page in reader.pages]
        return pages

    def create_embeddings(self, texts: List[str]):
        """Generates embeddings for a list of texts and stores them."""
        for i, text in enumerate(texts):
            response = self.aibrary.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float",
            )
            self.embeddings[i] = {"embedding": response.data[0].embedding, "text": text}

    def find_relevant_chunks(self, question: str, top_k=3) -> List[Dict]:
        """Finds the top-k most relevant chunks for a given question."""
        question_embedding = (
            self.aibrary.embeddings.create(
                model=self.embedding_model,
                input=question,
                encoding_format="float",
            )
            .data[0]
            .embedding
        )

        scores = {
            idx: self.cosine_similarity(question_embedding, chunk["embedding"])
            for idx, chunk in self.embeddings.items()
        }
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"text": self.embeddings[idx]["text"], "score": score}
            for idx, score in sorted_chunks[:top_k]
        ]

    def cosine_similarity(self, vec1, vec2) -> float:
        """Calculates the cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a**2 for a in vec1) ** 0.5
        norm2 = sum(b**2 for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)

    def ask_question(self, question: str) -> str:
        """Answers a question based on the most relevant chunks."""
        relevant_chunks = self.find_relevant_chunks(question)
        context = "\n".join(chunk["text"] for chunk in relevant_chunks)
        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        response = self.aibrary.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
