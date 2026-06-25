from pydantic_ai import Agent
from helpers.utils import get_prompt
from agents.models import LLM_MODEL


suggestions_agent = Agent(
    name="Suggestions Agent",
    model=LLM_MODEL,
    system_prompt=get_prompt('suggestions_system'),
    output_type=str,  # Plain text; parsed into list in app/tasks/suggestions.py
    retries=1,
)