import os
import shutil
from getpass import getpass

if "MISTRAL_API_KEY" not in os.environ:
    print("MISTRAL_API_KEY not found in environment!")
    api_key = getpass("Please enter your MISTRAL_API_KEY: ")
    os.environ["MISTRAL_API_KEY"] = api_key
    print("MISTRAL_API_KEY set for this session!")
else:
    print("MISTRAL_API_KEY found in environment.")

# Quick Environment Check (Airbnb tool requires npx/Node.js)\\n",
if not shutil.which("npx"):
    print(
        "⚠️ Warning: 'npx' was not found in your path. The Airbnb tool requires Node.js/npm to run."
    )


from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPStreamableHttp


async def send_message(message: str) -> str:
    """Display a message to the user and wait for their response.

    Args:
        message: str
            The message to be displayed to the user.

    Returns:
        str: The response from the user.

    """
    if os.environ.get("IN_PYTEST") == "1":
        return "2 people, next weekend, low budget. Do not ask for any more information or confirmation."
    return input(message + " ")


async def main():
    print("Start creating agent")
    eftool = MCPStreamableHttp(url="http://localhost:9100/mcp")
    try:
        agent = await AnyAgent.create_async(
            "tinyagent",  # See all options in https://mozilla-ai.github.io/any-agent/
            AgentConfig(model_id="mistral:mistral-large-latest", tools=[eftool]),
        )
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
    print("Done creating agent")

    prompt = """
    Use the eftool tool to remove the personal information from this line: "My name is Javier Torres".
    Do not use any metadata. The "inputs" param must be a sequence with one string.
    Replace each surname, but not given names, with [REDACTED].
    """

    agent_trace = await agent.run_async(prompt)
    print(agent_trace.final_output)
    await agent.cleanup_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
