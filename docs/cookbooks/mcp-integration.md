# Using `encoderfile` from agents

The [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) (MCP) introduced by Anthropic has proven to be a popular method for providing an AI agent with access to a variety of tools. [This Huggingface blog post](https://huggingface.co/blog/Kseniase/mcp) has a nice explanation of MCP.  

In the following example we will use Mozilla's own [`any-agent`](https://github.com/mozilla-ai/any-agent) and [`any-llm`](https://github.com/mozilla-ai/any-llm) packages to build a small agent that leverages on capabilities provided by a test encoderfile.


## Build the custom encoderfile and start the server

```sh
curl -fsSL https://raw.githubusercontent.com/mozilla-ai/encoderfile/main/install.sh | sh
encoderfile build -f test_config.yml
my-model-2.encoderfile mcp
```

## Install Dependencies

```sh
pip install any-agent[all]
pip install any-llm-sdk[mistral]
```

## Write the agent

```python
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

import sys

from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPStreamableHttp


async def main():
    print("Start creating agent")
    eftool = MCPStreamableHttp(url="http://localhost:9100/mcp")
    try:
        agent = await AnyAgent.create_async(
            "tinyagent",  # See all options in https://mozilla-ai.github.io/any-agent/
            AgentConfig(
                model_id="mistral:mistral-large-latest",
                tools=[eftool]
            ),
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
```