import requests
import streamlit as st


def chainInference():
    st.title("NEC LLM BOT")
    query = st.text_input("You:")
    if st.button("Send"):
        data = {
            "query": query,
        }
        response = requests.post('http://13.215.96.13:5000/chatbot', json=data)
        st.text("Chatbot: " + response.json()['response'])


if __name__ == "__main__":
    chainInference()
