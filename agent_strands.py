import os
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

# Read API key from env var (fail fast if missing)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Export it first, e.g.:\n"
        "  # macOS/Linux (zsh/bash)\n"
        "  export OPENAI_API_KEY=sk-... \n"
        "  # Windows PowerShell\n"
        "  $env:OPENAI_API_KEY='sk-...'\n"
    )

model = OpenAIModel(
    client_args={"api_key": api_key},
    model_id="gpt-4o",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    },
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2")
print(response)
