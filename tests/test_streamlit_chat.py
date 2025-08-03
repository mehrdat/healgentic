import streamlit as st
import lorem
import time
from random import randint


def chat_stream():
    for _ in range(randint(3, 9)):
        yield lorem.sentence() + " "
        time.sleep(0.5)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "".join(chat_stream())}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if user_input := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = "".join(chat_stream())
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})