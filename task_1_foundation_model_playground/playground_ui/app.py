from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools, AgentOutputParser, initialize_agent, ConversationalChatAgent, AgentExecutor
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
import streamlit as st
from sagemaker import session
from datetime import datetime
import json
import os
import re
import subprocess
from typing import Dict, List
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
import warnings
warnings.filterwarnings("ignore")


GLOBAL_TEMPERATURE = 0.6
GLOBAL_TOP_P = 0.7
GLOBAL_TOP_K = 30
app_path = os.path.dirname(os.path.realpath(__file__))


def format_messages(messages: List[Dict[str, str]]) -> List[str]:
    """
    Format messages for Llama 3+ chat models.
    
    The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and 
    alternating (u/a/u/a/u...). The last message must be from 'user'.
    """
    # auto assistant suffix
    # messages.append({"role": "assistant"})
    
    output = "<|begin_of_text|>"
    # Adding the inferred prefix
    _system_prefix = f"\n\nCutting Knowledge Date: December 2023\nToday Date: {datetime.now().strftime('%d %b %Y')}\n\n"
    for i, entry in enumerate(messages):
        output += f"<|start_header_id|>{entry['role']}<|end_header_id|>"
        if i == 0:
            output += f"{_system_prefix}{entry['content']}<|eot_id|>"
        elif i >= 1 and 'content' in entry:
            output += f"\n\n{entry['content']}<|eot_id|>"
    output += "<|start_header_id|>assistant<|end_header_id|>\n"
    return output


f = open(os.path.join(app_path, "endpoint_name.txt"), "r")
endpoint_name = f.read()
f.close()


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs):
        base_input = [
            {
                "role" : "system", 
                "content" : "You are a helpful ai assistant. You are polite and prioritize responses to a user in a friendly and helpful manner!"
            },
            {
                "role" : "user", 
                "content" : prompt
            }
        ]
        optz_input = format_messages(base_input)
        input_str = json.dumps({
            "inputs" : optz_input, 
            "parameters" : {**model_kwargs}
        })
        return input_str.encode('utf-8')
    
    def transform_output(self, output):
        response_json = json.loads(output.read().decode("utf-8"))
        print("response_json", response_json)
        return response_json["generated_text"]


st.set_page_config(page_title="Amazon SageMaker Workshop", page_icon="🦜")
st.image(os.path.join(app_path, "media", "banner.svg"), use_column_width=True)  
st.title("🦜 Amazon SageMaker & LangChain: Llama 3.1 8B Playground")
st.sidebar.image(os.path.join(app_path, "media", "logo.svg"), use_column_width=True) 

content_handler = ContentHandler()
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, 
    return_messages=True, 
    memory_key="history", 
    ai_prefix="AI Assistant"
)

llm=SagemakerEndpoint(
     endpoint_name=endpoint_name, 
     region_name=session.Session().boto_region_name, 
     model_kwargs={
         "max_new_tokens": 512, 
         "top_p": GLOBAL_TOP_P, 
         "top_k": GLOBAL_TOP_K,
         "temperature": GLOBAL_TEMPERATURE
     },
     content_handler=content_handler
 )


# Add sliders for Temperature, Top p, and Top k
st.sidebar.text("Adjust params and submit!")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, GLOBAL_TEMPERATURE, 0.1)
top_p = st.sidebar.slider("Top p", 0.0, 1.00, GLOBAL_TOP_P, 0.1)
top_k = st.sidebar.slider("Top k", 5, 50, GLOBAL_TOP_K, 5)

# Function to reinitialize the conversation agent with new parameters
def reinitialize_agent():
    llm.model_kwargs.update(
        {
            "temperature": temperature, 
            "top_p": top_p,
            "top_k": top_k
        }
    )
    memory.clear()
    msgs.clear()  # Clear StreamlitChatMessageHistory
    st.rerun()  # Rerun the script to reset the UI

# Submit button
if st.sidebar.button("Submit"):
    reinitialize_agent()

template = """The following is a friendly conversation between a human and an AI. The AI answers a user's question briefly and is only talkative when required.

Current conversation:
{history}

Human: {input}
AI Assistant:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

agent = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=memory
)

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    memory.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Please ask me a question!"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.predict(input=prompt, callbacks=[st_cb])
        # response = re.sub("\{.*?\}","",response)
        st.write(response)
