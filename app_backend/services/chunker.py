from typing import List, Dict
import uuid
import logging
import time

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
	"""
	Split text into overlapping chunks.
	Returns list of dicts: {"id": str, "text": str}
	"""
	chunk_start = time.time()
	logger.debug(f"chunk_text started, text length: {len(text)}, chunk_size: {chunk_size}, overlap: {overlap}")
	
	if not text:
		logger.warning("Empty text provided for chunking")
		return []

	try:
		tokens = text.split()
		logger.debug(f"Text split into {len(tokens)} tokens")
		
		chunks = []
		start = 0
		n = len(tokens)
		while start < n:
			end = start
			# accumulate words until approx chunk_size characters
			current = []
			length = 0
			while end < n and length < chunk_size:
				token = tokens[end]
				current.append(token)
				length += len(token) + 1
				end += 1

			chunk_text = " ".join(current).strip()
			chunks.append({"id": str(uuid.uuid4()), "text": chunk_text})

			# move start forward with overlap (guarantee progress to avoid infinite loops)
			overlap_words = max(0, overlap // 5)  # crude word-based overlap
			next_start = end - overlap_words
			if next_start <= start:
				next_start = end
			start = next_start

		total_time = time.time() - chunk_start
		logger.info(f"Text chunking completed in {total_time:.2f}s, created {len(chunks)} chunks")
		if chunks:
			logger.debug(f"First chunk size: {len(chunks[0]['text'])} chars, Last chunk size: {len(chunks[-1]['text'])} chars")
		
		return chunks
		
	except Exception as e:
		logger.error(f"Text chunking failed: {str(e)}")
		raise

