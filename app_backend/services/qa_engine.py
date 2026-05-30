from typing import List, Dict, Any
import logging
import os
import time

from app_backend.utils.vector_store import VectorStore
from app_backend.services.embedder import embed_texts
from app_backend.config import (
	OPENAI_API_KEY,
	OPENAI_CHAT_MODEL,
	OPENAI_MAX_CONTEXT_CHARS,
	OPENAI_MAX_OUTPUT_TOKENS,
)

logger = logging.getLogger(__name__)

from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
	global _client
	if _client is None:
		api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
		if not api_key:
			raise RuntimeError("OPENAI_API_KEY is not set")
		_client = OpenAI(api_key=api_key)
	return _client


class QAEngine:
	def __init__(self, store: VectorStore, gen_model: str = OPENAI_CHAT_MODEL):
		self.store = store
		self.gen_model = gen_model

	def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
		answer_start = time.time()
		logger.info(f"QAEngine.answer called with top_k={top_k}")
		logger.debug(f"Question: {question[:100]}...")
		
		try:
			logger.debug("Embedding question...")
			embed_start = time.time()
			q_emb = embed_texts([question])[0]
			embed_time = time.time() - embed_start
			logger.debug(f"Question embedded in {embed_time:.2f}s, dimension: {len(q_emb)}")
		
			logger.debug(f"Querying vector store for top {top_k} results...")
			query_start = time.time()
			hits = self.store.query(q_emb, top_k=top_k)
			query_time = time.time() - query_start
			logger.info(f"Vector store query completed in {query_time:.2f}s, found {len(hits)} results")

			context_parts = []
			sources = []
			for idx, (doc_id, score, meta) in enumerate(hits):
				text = meta.get("text", "")
				logger.debug(f"Hit {idx}: doc_id={doc_id}, score={score:.4f}, text_length={len(text)}")
				# Better context formatting
				context_parts.append(f"Source {idx + 1} (relevance score: {score:.2f}):\n{text}")
				sources.append({"id": doc_id, "text": text, "score": score})

			context = "\n".join(context_parts) if context_parts else ""
			if len(context) > OPENAI_MAX_CONTEXT_CHARS:
				context = context[:OPENAI_MAX_CONTEXT_CHARS] + "\n...[truncated]"
			logger.info(f"Context assembled, total length: {len(context)} characters, hits: {len(hits)}")
			if len(context) > 0:
				logger.debug(f"First 500 chars of context: {context[:500]}")

			client = _get_client()
			logger.debug("Calling OpenAI chat completion...")
			gen_start = time.time()
			
			user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
			logger.debug(f"User message length: {len(user_message)}, context included: {len(context) > 0}")
			
			resp = client.chat.completions.create(
				model=self.gen_model,
				messages=[
					{
						"role": "system",
						"content": (
							"You are a helpful assistant that answers questions based on the provided document context. "
							"Use the information from the provided sources to answer the user's question. "
							"If the answer cannot be found in the provided context, clearly state that the information is not available in the document. "
							"Be concise and cite the relevant source numbers when applicable."
						),
					},
					{
						"role": "user",
						"content": user_message,
					},
				],
				temperature=0,
				max_tokens=OPENAI_MAX_OUTPUT_TOKENS,
			)
			gen_time = time.time() - gen_start
			logger.info(f"OpenAI answer generated in {gen_time:.2f}s")

			answer_text = (resp.choices[0].message.content or "").strip()
			logger.debug(f"Answer text length: {len(answer_text)} characters")

			total_time = time.time() - answer_start
			logger.info(f"QAEngine.answer completed in {total_time:.2f}s with {len(sources)} sources")

			return {"answer": answer_text, "sources": sources}
			
		except Exception as e:
			logger.error(f"QAEngine.answer failed: {str(e)}", exc_info=True)
			raise


