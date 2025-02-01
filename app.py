import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def user_input(user_question):
    """Handles user input and retrieves responses from the conversation chain."""
    if st.session_state.conversation is None:
        st.warning("Please upload and process a document first.")
        return
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response.get('chat_history', [])

    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("User:", message.content)
        else:
            st.write("Reply:", message.content)

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(page_title="Information Retrieval")
    st.header("Document-Information-Retrieval-System")

    # Initialize session state variables only if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    # User input box
    user_question = st.text_input("What do you want to know from the document?")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your documents", accept_multiple_files=True)

        if st.button("Read this."):
            if not pdf_docs:
                st.warning("Please upload at least one document.")
            else:
                with st.spinner("Reading..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Done!")

if __name__ == "__main__":
    main()
