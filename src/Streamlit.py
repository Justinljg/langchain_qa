import streamlit as st
import requests
from PIL import Image


def main():

    st.set_page_config(
        page_title="Chatbot", page_icon=":robot_face:"
    )
    st.title("Document Q&A Chatbot :robot_face:")

    with st.sidebar:

        openai_api_key = st.text_input("OpenAI API key")

        temperature = st.slider("Temperature/Ranfomness", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

        st.header("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your pdf file here",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Upload"):

            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            elif not openai_api_key:
                st.warning("Please enter your OpenAI API key.")

            with st.spinner("Processing..."):
                pdf_docs = [("files", pdf.getvalue()) for pdf in pdf_docs]
                headers = {"OpenAI-API-Key": openai_api_key}
                response=requests.post(
                    "http://fastapi:8000/upload", files=pdf_docs, headers=headers
                )
                # Check the response from FastAPI
                if response.status_code == 200:
                    st.success("Files uploaded successfully.")

                else:
                    st.error(f"File upload failed with status code {response.status_code}.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if query := st.chat_input("Ask your questions"):
        # Display user message in chat message container
        st.chat_message("user").write(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        chat_data = {"query": query, "temperature": temperature}
        chat_response = requests.post("http://fastapi:8000/chat", json=chat_data)
        response = chat_response.json()

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
