## smart-doc-qa

FastAPI backend that:
- uploads a PDF, extracts text
- chunks + embeds chunks (OpenAI embeddings)
- answers questions using retrieved chunks (OpenAI chat)

## Setup (Windows PowerShell)

Create and activate a venv, then install deps:

```bash
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your OpenAI key (recommended) or copy `.env.example` to `.env`:

```bash
$env:OPENAI_API_KEY="YOUR_KEY"
```

Run the API:

```bash
uvicorn app_backend.main:app --reload --port 8000
```

## Endpoints

- `POST /qa/upload` (multipart form-data key: `file`)
- `POST /qa/ask` JSON body: `{ "question": "...", "top_k": 5 }`

## Low-cost model defaults

- **Embeddings**: `text-embedding-3-small`
- **Chat**: `gpt-4o-mini`
