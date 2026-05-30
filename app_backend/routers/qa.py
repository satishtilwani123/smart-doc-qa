from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi import Depends
from typing import List
import logging
import time
import traceback

from ..models.schema import UploadResponse, AskRequest, AnswerResponse
from ..services import parser, chunker, embedder, qa_engine
from ..utils.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

# create a single in-memory store for the app lifetime
store = VectorStore()
engine = qa_engine.QAEngine(store)


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
	start_time = time.time()
	logger.info(f"=== UPLOAD STARTED ===")
	logger.info(f"Filename: {file.filename}")
	logger.info(f"Content type: {file.content_type}")
	print(f"[UPLOAD] Started for {file.filename}")
	
	try:
		logger.debug("Reading file content...")
		content = await file.read()
		read_time = time.time()
		logger.info(f"File read completed in {read_time - start_time:.2f}s, size: {len(content)} bytes")
			
		if not content:
			logger.error("Empty file received")
			raise HTTPException(status_code=400, detail="Empty file")

		logger.debug("Starting PDF parsing...")
		parse_start = time.time()
		text = parser.parse_pdf(content)
		parse_time = time.time()
		print(text[:500])  # Print the first 500 characters of the extracted text for debugging
		logger.info(f"PDF parsing completed in {parse_time - parse_start:.2f}s")
		logger.info(f"Extracted text length: {len(text)} characters")
		print(f"[UPLOAD] Parsed text length: {len(text)} chars")
			
		if not text or len(text.strip()) == 0:
			logger.warning("No text extracted from PDF")
			logger.warning(f"First 200 bytes of content: {content[:200]}")
			
		logger.debug("Starting text chunking...")
		chunk_start = time.time()
		chunks = chunker.chunk_text(text)
		chunk_time = time.time()
		logger.info(f"Text chunking completed in {chunk_time - chunk_start:.2f}s")
		logger.info(f"Created {len(chunks)} chunks")
		print(f"[UPLOAD] Created {len(chunks)} chunks")

		if not chunks:
			logger.warning("No chunks created from text")
			return UploadResponse(success=False, chunks_created=0)

		texts = [c["text"] for c in chunks]
		ids = [c["id"] for c in chunks]
		metadatas = [{"text": t} for t in texts]

		print(f"[UPLOAD] DEBUG - texts: {len(texts)}, ids: {len(ids)}, metadatas: {len(metadatas)}")
		logger.debug(f"Starting embedding generation for {len(texts)} chunks...")
		print(f"[UPLOAD] DEBUG - About to call embedder.embed_texts()...")
		embedd_start = time.time()
		vectors = embedder.embed_texts(texts)
		embedd_time = time.time()
		print(f"[UPLOAD] DEBUG - Embedding completed, vectors: {len(vectors)}")
		logger.info(f"Embedding generation completed in {embedd_time - embedd_start:.2f}s")
		logger.info(f"Generated {len(vectors)} embeddings")
		if vectors:
			logger.debug(f"Embedding dimension: {len(vectors[0])}")
		print(f"[UPLOAD] Embeddings complete, adding to store...")
			
		logger.debug("Adding embeddings to vector store...")
		store_start = time.time()
		print(f"[UPLOAD] DEBUG - About to call store.add()...")
		store.add(ids, vectors, metadatas)
		store_time = time.time()
		print(f"[UPLOAD] DEBUG - Store add completed")
		logger.info(f"Vector store updated in {store_time - store_start:.2f}s with {len(chunks)} chunks")
		print(f"[UPLOAD] Store updated with {len(chunks)} chunks")
			
		total_time = time.time() - start_time
		logger.info(f"=== UPLOAD COMPLETED ===")
		logger.info(f"Total time: {total_time:.2f}s")
		logger.info(f"Breakdown - Read: {read_time-start_time:.2f}s, Parse: {parse_time-parse_start:.2f}s, Chunk: {chunk_time-chunk_start:.2f}s, Embed: {embedd_time-embedd_start:.2f}s, Store: {store_time-store_start:.2f}s")

		return UploadResponse(success=True, chunks_created=len(chunks))
		
	except Exception as e:
		logger.error(f"=== UPLOAD FAILED ===")
		logger.error(f"Exception: {str(e)}")
		logger.error(f"Traceback: {traceback.format_exc()}")
		print(f"[UPLOAD] ERROR: {str(e)}")
		return UploadResponse(success=False, chunks_created=0)

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: AskRequest):
	start_time = time.time()
	logger.info(f"=== ASK STARTED ===")
	logger.info(f"Question: {request.question}")
	logger.info(f"Top-k: {request.top_k}")
	print(f"[ASK] Question: {request.question}")
	
	try:
		print(f"[ASK] Processing (this may take a moment on first run while loading model)...")
		resp = engine.answer(request.question, top_k=request.top_k)
		process_time = time.time() - start_time
		logger.info(f"Answer generated in {process_time:.2f}s with {len(resp['sources'])} sources")
		print(f"[ASK] Answer generated")
		logger.info(f"=== ASK COMPLETED ===")
		
		return AnswerResponse(answer=resp["answer"], sources=resp["sources"])
	except Exception as e:
		logger.error(f"=== ASK FAILED ===")
		logger.error(f"Exception: {str(e)}")
		logger.error(f"Traceback: {traceback.format_exc()}")
		raise
