import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
from helpers.utils import get_logger

load_dotenv()
logger = get_logger(__name__)

# Get configurations from environment variables
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama').lower()
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'qwen2.5:7b')

logger.info(f"LLM_PROVIDER loaded: '{LLM_PROVIDER}'")
logger.info(f"LLM_MODEL_NAME loaded: '{LLM_MODEL_NAME}'")

if LLM_PROVIDER == 'ollama':
    LLM_MODEL = OpenAIModel(
        LLM_MODEL_NAME,
        provider=OpenAIProvider(
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434') + '/v1',
            api_key="ollama",  # Ollama accepts any non-empty string
        ),
    )
elif LLM_PROVIDER == 'openrouter':
    LLM_MODEL = OpenAIModel(
        LLM_MODEL_NAME,
        provider=OpenAIProvider(
            base_url='https://openrouter.ai/api/v1',
            api_key=os.getenv('OPENROUTER_API_KEY'),
        ),
    )
else:
    raise ValueError(
        f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. "
        f"Must be one of: 'ollama', 'openrouter'"
    )
