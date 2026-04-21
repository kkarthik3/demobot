import streamlit as st
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_aws import ChatBedrockConverse
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon=":robot_face:",
    layout="wide"
)

# -------------------------------
# Model Configurations
# -------------------------------
AWS_MODELS = {
    "Claude 3.5 Haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "Llama 3.3 70B": "us.meta.llama3-3-70b-instruct-v1:0"
}

GROQ_MODELS = {
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "LLaMA3 70B": "llama-3.3-70b-versatile",
}

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.title("Chat Configuration")

    provider = st.selectbox("Select Provider", ["AWS Bedrock", "Groq"])

    if provider == "AWS Bedrock":
        selected_model = st.selectbox("Select Model", list(AWS_MODELS.keys()))
        model_id = AWS_MODELS[selected_model]
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1"])
        max_tokens = None
    else:
        selected_model = st.selectbox("Select Model", list(GROQ_MODELS.keys()))
        model_id = GROQ_MODELS[selected_model]
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        max_tokens = st.number_input("Max Tokens", min_value=1, step=1, value=512)
        region = None

    api_password = st.text_input("Enter API Password", type="password")
    stored_password = os.getenv("API_PASSWORD")

    system_message = st.text_area("System Message", value="", height=100)

# -------------------------------
# Session State Initialization
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "current_model" not in st.session_state:
    st.session_state.current_model = None

# Reset if model changes
if st.session_state.current_model != model_id:
    st.session_state.initialized = False
    st.session_state.current_model = model_id

# -------------------------------
# Authentication + LLM Setup
# -------------------------------
if api_password and api_password == stored_password:

    if not st.session_state.initialized:
        try:
            if provider == "AWS Bedrock":
                llm = ChatBedrockConverse(
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name=region,
                )
            else:
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
            st.error(f"Error initializing model: {e}")

    # -------------------------------
    # System Message Handling
    # -------------------------------
    if len(st.session_state.messages) == 0 or (
        isinstance(st.session_state.messages[0], SystemMessage) and
        st.session_state.messages[0].content != system_message
    ):
        st.session_state.messages = [SystemMessage(content=system_message)]

    # -------------------------------
    # Chat UI
    # -------------------------------
    st.title("AI Chat Assistant")

    for message in st.session_state.messages:
        if not isinstance(message, SystemMessage):
            with st.chat_message(message.type):
                st.markdown(message.content)

    # -------------------------------
    # User Input
    # -------------------------------
    if prompt := st.chat_input("Type your message here..."):
        if prompt.strip():
            st.session_state.messages.append(HumanMessage(content=prompt))

            with st.chat_message("user"):
                st.markdown(prompt)

            # -------------------------------
            # Assistant Response (FIXED)
            # -------------------------------
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.llm.invoke(st.session_state.messages)

                        # Safe usage of response
                        if response and hasattr(response, "content"):
                            st.markdown(response.content)
                            st.session_state.messages.append(
                                AIMessage(content=response.content)
                            )
                        else:
                            st.error("Received empty response from model.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

else:
    st.warning("Please enter the correct API password to access the chat.")
