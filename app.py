# Chatbot to chat with the LangChain docs!
# Uses ChatGPT, LangChain, Chroma, FastAPI, Streamlit
# This file contains the code for the Streamlit application.

# Note: Had to downgrade streamlit from 1.35.0 to 1.32.0 to get the spinner to work properly;
# see https://discuss.streamlit.io/t/repeated-assistant-answer-when-spinning/69129

import streamlit as st
import requests

# Set Streamlit title
st.title("LangChain Docs Chatbot")

# Set avatars for assistant and user
assistant_avatar = "🦜️"
user_avatar = "👤"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything about LangChain!"}]

# Print chat history with appropriate avatar
for message in st.session_state.messages:
    if message["role"] == "assistant":
        avatar = assistant_avatar
    else:
        avatar = user_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

# Define prompt for user
if prompt := st.chat_input("Enter your question"):
    with st.chat_message("user", avatar=user_avatar):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant", avatar=assistant_avatar):
        with st.spinner("Please wait..."):
                json_response = requests.post(
                    url="https://27ab-2601-147-4800-32f0-1d59-565f-9095-6564.ngrok-free.app/invoke",
                    json={"input": {"question": prompt}}
                ).json()
                response = json_response["output"]["answer"]
                st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
