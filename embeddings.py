from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

def save_embed_data(text_chunks, api_key, db_location,embed_model_name):
    """
    Desc:
    Save embedded text to FAISS (Facebook AI Similarity Search) vector data store.

    Args:
    text_chunks (List[str]) -> contains list of data chunks that required embedding
    api_key (str) -> Target embed model api key
    db_location (str) -> FAISS local vector data store location
    embed_model_name (str) -> Name of the embed model

    Return:
     None
    
    """
    embeddings = MistralAIEmbeddings(model=embed_model_name,api_key=api_key)
    embed_data = FAISS.from_texts(text_chunks, embedding=embeddings)
    embed_data.save_local(db_location)

def get_embed_data(api_key,embed_model_name,db_location):
    """
    Desc:
    Fetch the stored embed vector data

    Args:
    api_key (str) -> Target chat model api key
    embed_model_name (str) -> Name of the chat model
    db_location (str) -> Local FAISS vector store location

    Return:
     FAISS
    """
    embeddings = MistralAIEmbeddings(model=embed_model_name,api_key=api_key)
    db = FAISS.load_local(db_location, embeddings,allow_dangerous_deserialization=True)
    return db

    