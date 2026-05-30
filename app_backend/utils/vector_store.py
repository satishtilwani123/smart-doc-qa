from typing import List, Dict, Any, Tuple
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

try:
	import faiss
	logger.debug("FAISS library imported successfully")
except Exception as e:
	logger.warning(f"FAISS library not available: {str(e)}, will use numpy-based vector store")
	faiss = None


class VectorStore:
	def __init__(self, dim: int = None):
		self.ids: List[str] = []
		self.metadatas: List[Dict[str, Any]] = []
		self._dim = dim
		self._np_index: List[np.ndarray] = []
		self._faiss_index = None

	def _ensure_faiss(self, dim: int):
		if faiss is None:
			return False
		if self._faiss_index is None:
			self._faiss_index = faiss.IndexFlatIP(dim)
		return True

	def add(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
		add_start = time.time()
		logger.debug(f"VectorStore.add started, vectors count: {len(vectors)}")
		
		if not vectors:
			logger.warning("Empty vectors list provided to VectorStore.add")
			return
		
		try:
			dim = len(vectors[0])
			if self._dim is None:
				self._dim = dim
				logger.info(f"Vector dimension set to {dim}")
			elif self._dim != dim:
				logger.warning(f"Dimension mismatch: expected {self._dim}, got {dim}")

			logger.debug("Converting vectors to numpy array...")
			vecs_np = np.array(vectors, dtype=np.float32)
			logger.debug(f"Numpy array shape: {vecs_np.shape}")
			
			logger.debug("Normalizing vectors...")
			# normalize for cosine similarity when using inner product
			norms = np.linalg.norm(vecs_np, axis=1, keepdims=True) + 1e-10
			vecs_np = vecs_np / norms
			logger.debug("Vector normalization completed")

			if self._ensure_faiss(dim):
				logger.debug(f"Adding {len(vectors)} vectors to FAISS index...")
				self._faiss_index.add(vecs_np)
				logger.debug(f"FAISS index size now: {self._faiss_index.ntotal}")
			else:
				logger.debug(f"Adding {len(vectors)} vectors to numpy index...")
				self._np_index.append(vecs_np)
				logger.debug(f"Numpy index count: {len(self._np_index)}")

			logger.debug("Adding metadata...")
			for i, _ in enumerate(vectors):
				self.ids.append(ids[i])
				self.metadatas.append(metadatas[i])
			
			add_time = time.time() - add_start
			logger.info(f"VectorStore.add completed in {add_time:.2f}s, total vectors: {len(self.ids)}")
			
		except Exception as e:
			logger.error(f"VectorStore.add failed: {str(e)}")
			raise

	def query(self, vector: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
		query_start = time.time()
		logger.debug(f"VectorStore.query started, top_k={top_k}, vector dimension: {len(vector)}")
		
		if self._dim is None:
			logger.warning("Vector store is empty, returning empty results")
			return []
		
		try:
			logger.debug("Normalizing query vector...")
			q = np.array(vector, dtype=np.float32)
			q = q / (np.linalg.norm(q) + 1e-10)

			results = []
			if self._faiss_index is not None:
				logger.debug(f"Searching FAISS index with {self._faiss_index.ntotal} vectors...")
				search_start = time.time()
				D, I = self._faiss_index.search(np.expand_dims(q, axis=0), top_k)
				search_time = time.time() - search_start
				I = I[0]
				D = D[0]
				logger.debug(f"FAISS search completed in {search_time:.3f}s")
				
				valid_results = 0
				for idx, score in zip(I, D):
					if idx < 0:
						logger.debug(f"Skipping invalid index: {idx}")
						continue
					valid_results += 1
					results.append((self.ids[int(idx)], float(score), self.metadatas[int(idx)]))
				logger.info(f"FAISS search returned {valid_results} valid results")
				return results

			# fallback to numpy brute force
			if self._np_index:
				logger.debug(f"Using numpy fallback with {len(self._np_index)} index chunks...")
				brute_start = time.time()
				mat = np.vstack(self._np_index)
				logger.debug(f"Matrix shape: {mat.shape}")
				mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
				q_norm = q
				scores = (mat_norm @ q_norm).astype(float)
				idxs = np.argsort(-scores)[:top_k]
				brute_time = time.time() - brute_start
				logger.debug(f"Numpy brute force search completed in {brute_time:.3f}s")
				
				for i in idxs:
					results.append((self.ids[int(i)], float(scores[int(i)]), self.metadatas[int(i)]))
				
				logger.info(f"Numpy search returned {len(results)} results")
				return results
			
			logger.warning("No index available, returning empty results")
			return []
			
		except Exception as e:
			logger.error(f"VectorStore.query failed: {str(e)}", exc_info=True)
			raise


