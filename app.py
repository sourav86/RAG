import streamlit as st
import os
import documents_transformation as dt
import embeddings as eb
import documents_retrievals as dr
import prompts
from dotenv import load_dotenv

load_dotenv()

MISTRAL_CHAT_MODEL = "mistral-small"
MISTRAL_EMBEDDING_MODEL = "mistral-embed"
DB_LOCATION = "vectorstore\db_faiss"
API_KEY = os.getenv('MISTRAL_API_KEY')

st.set_page_config(page_title="Med Book Assistant", layout="wide")
def main():
    st.header("AI medbook chatbot")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and API_KEY:  # Ensure API key and user question are provided
        store_med_data = eb.get_embed_data(api_key=API_KEY,
                                           embed_model_name=MISTRAL_EMBEDDING_MODEL,
                                           db_location=DB_LOCATION)
        response = dr.user_input(user_question=user_question, 
                                 api_key=API_KEY,
                                 db=store_med_data,
                                 prompt_template= prompts.prompt_template,
                                 chat_model_name=MISTRAL_CHAT_MODEL)
        st.write("Reply: ", response["output_text"])

    with st.sidebar:
        pdf_docs = st.file_uploader("Upload pdf files for medical book", accept_multiple_files=True, key="pdf_uploader")
        has_sumitted = st.button("Submit & Process", key="process_button")
        if has_sumitted:
            with st.spinner("Processing..."):
                raw_text = dt.get_pdf_text(pdf_docs)
                text_chunks = dt.get_text_chunks(raw_text)
                eb.save_embed_data(text_chunks=text_chunks, 
                                    api_key=API_KEY,
                                    db_location=DB_LOCATION,
                                    embed_model_name=MISTRAL_EMBEDDING_MODEL)
                st.success("Done")

if __name__ == "__main__":
    main()