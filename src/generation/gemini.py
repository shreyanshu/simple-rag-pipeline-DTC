from google import genai
from src.config import GOOGLE_API_KEY, GEMINI_MODEL


class GeminiGenerator:
    def __init__(self, model=GEMINI_MODEL):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = model

    def answer(self, question, matches):
        context_parts = []
        for i, match in enumerate(matches, start=1):
            md = match.metadata or {}
            context_parts.append(
                f"[Source {i} | {md.get('source')} | page {md.get('page')} | "
                f"score {match.score:.4f}]\n{md.get('text', '')}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful question-answering assistant.

Use ONLY the context below to answer the question.
If the context does not contain enough information, say:
"I couldn't find that information in the provided document."

Do not invent facts.

Context:
----------------
{context}
----------------

Question:
{question}

Give a concise answer and cite the relevant page numbers in the form [Page X].
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text
