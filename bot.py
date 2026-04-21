import streamlit as st
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_aws import ChatBedrockConverse
from langchain_groq import ChatGroq
from uuid import UUID

st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon=":robot_face:",
    layout="wide"
)

# Define available models for each provider
AWS_MODELS = {
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "llama 3.3 70b" : "us.meta.llama3-3-70b-instruct-v1:0"
}

GROQ_MODELS = {
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "LLaMA2 70B": "llama-3.3-70b-versatile",
}

with st.sidebar:
    st.title("Chat Configuration")
    
    # Model provider selection
    provider = st.selectbox("Select Provider", ["AWS Bedrock", "Groq"])
    
    # Model selection based on provider
    if provider == "AWS Bedrock":
        selected_model = st.selectbox("Select Model", list(AWS_MODELS.keys()), index=0)
        model_id = AWS_MODELS[selected_model]
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1"])
    else:  # Groq
        selected_model = st.selectbox("Select Model", list(GROQ_MODELS.keys()), index=0)
        model_id = GROQ_MODELS[selected_model]
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        max_tokens = st.number_input("Max Tokens", value=None, min_value=1, step=1)
    
    # Password input for API keys
    api_password = st.text_input("Enter API Password", type="password")
    stored_password = os.getenv("API_PASSWORD")
    
    # System message configuration
    system_message = st.text_area(
        "System Message",
        value="",
        height=100
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None

# Reset initialization if model changes
if st.session_state.current_model != model_id:
    st.session_state.initialized = False
    st.session_state.current_model = model_id

# Verify password and initialize LLM
if api_password and api_password == stored_password:
    if not st.session_state.initialized:
        if provider == "AWS Bedrock":
            try:
                llm = ChatBedrockConverse(
                    model=model_id,
                    temperature=temperature,
                    max_tokens=None,
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name=region,
                )
                st.session_state.llm = llm
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"Error initializing AWS Bedrock: {e}")
        else:
            try:
                llm = ChatGroq(
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=None,
                    max_retries=2,
                    api_key=os.getenv("GROQ_API_KEY")
                )
                st.session_state.llm = llm
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"Error initializing Groq: {e}")
    
    # Update system message if changed
    if len(st.session_state.messages) == 0 or (
        isinstance(st.session_state.messages[0], SystemMessage) and 
        st.session_state.messages[0].content != system_message
    ):
        st.session_state.messages = [SystemMessage(content=system_message)]
    
    # Main chat interface
    st.title("AI Chat Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        if not isinstance(message, SystemMessage):
            with st.chat_message(message.type):
                st.markdown(message.content)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        if prompt.strip():
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.llm.invoke(st.session_state.messages)
                        st.markdown(response.content)
                        st.session_state.messages.append(AIMessage(content=response.content))
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                    st.markdown(response.content)
                    st.session_state.messages.append(AIMessage(content=response.content))
                except Exception as e:
                    st.error(f"An error occurred: {e}")

else:
    st.warning("Please enter the correct API password to access the chat.")
