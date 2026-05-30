from io import BytesIO
from typing import List
import logging
import time
import re

logger = logging.getLogger(__name__)

try:
	from PyPDF2 import PdfReader
except Exception:
	PdfReader = None


def clean_extracted_text(text: str) -> str:
	"""
	Clean up PDF extraction artifacts like excessive spaces between characters.
	Converts "S e n i o r" → "Senior"
	"""
	# Remove spaces between single characters (character-by-character spacing issue)
	text = re.sub(r'(\w)\s(?=\w)', r'\1', text)
	
	# Fix multiple spaces that may have resulted
	text = re.sub(r' {2,}', ' ', text)
	
	return text.strip()


def parse_pdf(file_bytes: bytes) -> str:
	"""
	Extract raw text from PDF bytes.
	Falls back to simple byte->string decode if PyPDF2 is not available.
	"""
	parse_start = time.time()
	logger.debug(f"parse_pdf started, file size: {len(file_bytes)} bytes")
	
	if PdfReader is None:
		logger.warning("PyPDF2 not available, using fallback byte decoding")
		try:
			result = file_bytes.decode("utf-8", errors="ignore")
			logger.debug(f"Fallback decoding extracted {len(result)} characters")
			return result
		except Exception as e:
			logger.error(f"Fallback decoding failed: {str(e)}")
			return ""

	try:
		logger.debug("Creating BytesIO stream and PDF reader...")
		stream = BytesIO(file_bytes)
		reader = PdfReader(stream)
		logger.info(f"PDF reader created, pages: {len(reader.pages)}")
		
		texts: List[str] = []
		for idx, page in enumerate(reader.pages):
			try:
				page_start = time.time()
				page_text = page.extract_text() or ""
				page_time = time.time() - page_start
				texts.append(page_text)
				logger.debug(f"Page {idx} extracted in {page_time:.3f}s, length: {len(page_text)} chars")
			except Exception as e:
				logger.warning(f"Failed to extract text from page {idx}: {str(e)}")
				continue
		
		result = "\n".join(texts)
		result = clean_extracted_text(result)
		total_time = time.time() - parse_start
		logger.info(f"PDF parsing completed in {total_time:.2f}s, total text length: {len(result)} chars")
		return result
		
	except Exception as e:
		logger.error(f"PDF parsing failed: {str(e)}")
		return ""

