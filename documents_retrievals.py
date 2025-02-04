from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.vectorstores import FAISS

def get_conversational_chain(prompt_template,api_key,chat_model_name):
    """
    Desc:
    Get question and answer prompt chain.

    Args:
    prompt_template (str) -> prompt template used for conversation
    api_key (str) -> Target chat model api key
    chat_model_name (str) -> Name of the chat model

    Return:
     BaseCombineDocumentsChain
    """
    model = ChatMistralAI(model_name=chat_model_name, temperature=0.3, api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, api_key,db:FAISS,prompt_template,chat_model_name):
    """
    Desc:
    User input used for conversation

    Args:
    user_question (str) -> prompt template used for conversation
    api_key (str) -> Target chat model api key
    db (FAISS) -> FAISS db instance
    prompt_template (str) -> Prompt template for chat conversation
    chat_model_name (str) -> Name of the chat model

    Return:
     Dict[str,Any]
    """
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain(prompt_template=prompt_template,api_key=api_key,chat_model_name=chat_model_name)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response