import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _env(name: str, default: str | None = None) -> str | None:
	return os.getenv(name, default)


def _load_dotenv_fallback(dotenv_path: Path) -> None:
	try:
		with dotenv_path.open("r", encoding="utf-8") as fp:
			for raw_line in fp:
				line = raw_line.strip()
				if not line or line.startswith("#"):
					continue
				if "=" not in line:
					continue
				key, value = line.split("=", 1)
				key = key.strip()
				value = value.strip()
				if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
					value = value[1:-1]
				if key and key not in os.environ:
					os.environ[key] = value
	except Exception as exc:
		logger.warning("Failed to parse .env file %s: %s", dotenv_path, exc)


# Load a root .env file if present (keeps local dev simple on Windows)
root_env = Path(__file__).resolve().parents[1] / ".env"
if root_env.exists():
	try:
		from dotenv import load_dotenv
	except ImportError:
		logger.warning("python-dotenv is not installed; falling back to manual .env parser")
		_load_dotenv_fallback(root_env)
	else:
		load_dotenv(dotenv_path=root_env)
		logger.info("Loaded root .env from %s", root_env)

OPENAI_API_KEY: str | None = _env("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL: str = _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")  # type: ignore[assignment]
OPENAI_CHAT_MODEL: str = _env("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # type: ignore[assignment]

if not OPENAI_API_KEY:
	logger.warning("OPENAI_API_KEY is not set. Define it in the root .env or environment.")

# Guardrails to keep token usage/cost low
OPENAI_MAX_CONTEXT_CHARS: int = int(_env("OPENAI_MAX_CONTEXT_CHARS", "8000") or "8000")
OPENAI_MAX_OUTPUT_TOKENS: int = int(_env("OPENAI_MAX_OUTPUT_TOKENS", "250") or "250")
