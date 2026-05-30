from typing import List, Optional
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
	success: bool
	chunks_created: int = 0

class AskRequest(BaseModel):
	question: str = Field(..., example="What is the main point of document X?")
	top_k: int = Field(5, ge=1, le=50)

class SourceDocument(BaseModel):
	id: str
	text: str
	score: float

class AnswerResponse(BaseModel):
	answer: str
	sources: List[SourceDocument] = []