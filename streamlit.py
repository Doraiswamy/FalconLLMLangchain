import requests
import streamlit as st


def chainInference():
    st.title("NEC LLM BOT")
    query = st.text_input("You:")
    if st.button("Send"):
        data = {
            "prompt": query,
        }
        response = requests.post('http://13.215.96.13:5000/chatbot', json=data, timeout=50)
        st.text("Chatbot: " + response.json()['response'])


if __name__ == "__main__":
    chainInference()
