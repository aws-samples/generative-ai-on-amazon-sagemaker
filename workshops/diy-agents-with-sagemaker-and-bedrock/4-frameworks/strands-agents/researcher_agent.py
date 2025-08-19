import os
import boto3
import json

# Strands imports
from strands.models.bedrock import BedrockModel
from strands import Agent, tool
from strands_tools import retrieve, http_request


# AgentCore imports
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore_starter_toolkit import Runtime

# Get agent ARNs from deployed agents
kb_agent_arn = os.environ["KB_AGENT_ARN"]
web_agent_arn = os.environ["WEB_AGENT_ARN"]
region = os.environ["AWS_REGION"]
model = BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0")

print(f"KB Agent ARN: {kb_agent_arn}")
print(f"Web Agent ARN: {web_agent_arn}")

# Create AgentCore client
agentcore_client = boto3.client('bedrock-agentcore', region_name=region)


def invoke_agentcore_agent(agent_arn: str, query: str, agent_type: str = "agent") -> str:
    """Helper function to invoke AgentCore agents with consistent error handling"""
    try:
        response = agentcore_client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            qualifier="DEFAULT",
            payload=json.dumps({"prompt": query}),
            contentType="application/json",
            accept="application/json"
        )

        content = ''.join(chunk.decode('utf-8') for chunk in response.get("response", []))
        result = json.loads(content)
        if isinstance(result, dict) and 'result' in result:
            result_data = result['result']

            if isinstance(result_data, dict) and 'content' in result_data:
                if isinstance(result_data['content'], list):
                    # Extract text from content array
                    text_parts = []
                    for item in result_data['content']:
                        if isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                    return ''.join(text_parts)
                else:
                    return str(result_data['content'])
            else:
                return str(result_data)
        else:
            return str(result)
        
    except Exception as e:
        return f"Error searching {agent_type}: {str(e)}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for information about AgentCore, AWS services, and technical documentation.
    
    Args:
        query: The search query for the knowledge base
        
    Returns:
        str: Information retrieved from the knowledge base
    """
    return invoke_agentcore_agent(kb_agent_arn, f"{query}", "knowledge base")


@tool
def search_web(query: str) -> str:
    """Search the web for current information, trends, and real-time data.
    
    Args:
        query: The web search query
        
    Returns:
        str: Information retrieved from web sources
    """
    return invoke_agentcore_agent(web_agent_arn, f"{query}", "web search")


# Create the research agent
research_agent = Agent(
    model=model,
    tools=[search_knowledge_base, search_web],
    system_prompt="""You are an intelligent research agent that helps users find and correlate information from multiple sources.

You have access to two specialized tools:
1. search_knowledge_base: For internal documentation, AWS services info, and technical knowledge
2. search_web: For current information, trends, and real-time data

Research Strategy:
- Always synthesize information from both sources using search_knowledge_base and search_web tools
- Provide clear, well-structured responses with source attribution (URLs where available)
- Always add citations to your answers

Be thorough in your research and provide detailed, accurate answers."""
)


print("\nResearch Agent created successfully!")
print("Available tools: search_knowledge_base, search_web")
print("\nExample usage:")
print("result = research_agent('What is AgentCore and how does it compare to current AI agent platforms?')")
print("print(result.message)")


# BEDROCK AGENTCORE APP INITIALIZATION:
# BedrockAgentCoreApp is the primary SDK class for deploying agents to AgentCore Runtime.
# It provides the framework-agnostic foundation for any agent implementation.
app = BedrockAgentCoreApp()


# AGENTCORE ENTRYPOINT DECORATOR PATTERN:
# The @app.entrypoint decorator is the key integration point for AgentCore deployment.
# This transforms your agent function into an HTTP service compatible with Amazon Bedrock.
@app.entrypoint
def agent_entrypoint(payload):
    """
    Invoke the agent with the payload
    
    AGENTCORE PAYLOAD HANDLING:
    - AgentCore Runtime passes requests as payload dictionaries
    - Standard pattern extracts user input from payload["prompt"] 
    - Return values are automatically formatted for HTTP responses
    """
    user_input = payload.get("prompt")
    print("User input:", user_input)
    response = research_agent(user_input)
    return response.message['content'][0]['text']


# AGENTCORE LOCAL DEVELOPMENT SUPPORT:
# The app.run() method enables local testing before AgentCore deployment.
# This supports the rapid prototyping to production workflow.
if __name__ == "__main__":
    app.run()
