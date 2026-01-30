# Using `encoderfile` from agents

The [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) (MCP) introduced by Anthropic has proven to be a popular method for providing an AI agent with access to a variety of tools. [This Huggingface blog post](https://huggingface.co/blog/Kseniase/mcp) has a nice explanation of MCP.  

In the following example we will use Mozilla's own [`any-agent`](https://github.com/mozilla-ai/any-agent) and [`any-llm`](https://github.com/mozilla-ai/any-llm) packages to build a small agent that leverages on capabilities provided by a test encoderfile.


## Build the custom encoderfile and start the server

We will use the existing test config to build an encoderfile using one of the test models by Mozilla.ai. It will detect Personally Identifiable Information (PII) and tag it accordingly, using tags like `B-SURNAME` for, well, surnames, and `O` for non-PII tokens. As we will see, even if the output consists of logits and tags, the underlying LLM is usually robust enough to focus only on the tags and act appropriately.

```sh
curl -fsSL https://raw.githubusercontent.com/mozilla-ai/encoderfile/main/install.sh | sh
encoderfile build -f test_config.yml
```

After building it, we only need to set it up in MCP mode so it will listen to requests. By default it will bind to all interfaces, using port 9100. 

```sh
my-model-2.encoderfile mcp
```

## Install Dependencies

For this test, we will need the `any-agent` and `any-llm` Python packages:

```sh
pip install any-agent
pip install any-llm-sdk[mistral]
```

## Write the agent

Now we will write an agent with the appropriate prompt. We instruct the agent to use the provided tool, since the current description is fairly generic, and not use metadata that it might consider useful but is not documented anywhere in the tool itself. We will also instruct it to replace only surnames to showcase that the tags can be extracted appropriately:

```python
--8<-- "examples/agent_mcp_integration/agent_test.py"
```

After some struggling with the call conventions, the LLM finally obtains the information from the encoderfile and acts accordingly:

> `My name is Javier [REDACTED]`
