from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("LangChain Udemy Course- Documentation Helper Bot")

# Initialize the session state for the prompt if it does not already exist
if 'prompt' not in st.session_state:
    st.session_state.prompt = ""

# Create a text input that uses the session state to store the current prompt
prompt = st.text_input("Prompt", value=st.session_state.prompt, placeholder="Enter your prompt here..", key="prompt_input")

# Add a button for sending the message. When clicked, it will process the prompt.
send_button = st.button("Send")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if send_button and prompt:
    with st.spinner("Generating response.."):
        generated_response = run_llm(query=prompt)
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        
        # Clear the chat input box by resetting the session state value for prompt
        st.session_state.prompt = ""

if st.session_state["chat_answers_history"]:
    # Reverse the lists
    reversed_chat_answers_history = st.session_state["chat_answers_history"][::-1]
    reversed_user_prompt_history = st.session_state["user_prompt_history"][::-1]

    for generated_response, user_query in zip(
        reversed_chat_answers_history,
        reversed_user_prompt_history,
    ):
        message(user_query, is_user=True)
        message(generated_response)
