from typing import List
import logging
import os
import time

from openai import OpenAI

from ..config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
	global _client
	if _client is None:
		api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
		if not api_key:
			raise RuntimeError("OPENAI_API_KEY is not set")
		_client = OpenAI(api_key=api_key)
	return _client


def embed_texts(texts: List[str], model_name: str | None = None) -> List[List[float]]:
	"""Return embeddings for a list of texts using OpenAI embeddings."""
	embed_start = time.time()
	model = model_name or OPENAI_EMBEDDING_MODEL

	if not texts:
		logger.warning("Empty texts list provided for embedding")
		return []

	try:
		client = _get_client()
		resp = client.embeddings.create(model=model, input=texts)
		# OpenAI returns data in the same order as inputs.
		result = [d.embedding for d in resp.data]

		total_time = time.time() - embed_start
		if result:
			logger.info(
				f"OpenAI embedding generation completed in {total_time:.2f}s, count={len(result)}, dim={len(result[0])}, model={model}"
			)
		return result
	except Exception as e:
		logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
		raise


