# About the solution:
The Medical Document Assistant Bot aims to support healthcare professionals by providing precise answers to disease-related queries. Leveraging the comprehensive knowledge from curated medical literature, the bot facilitates efficient and accurate information retrieval. The solution integrates Mistral Large Language Model (LLM) with FAISS vector data store, implementing Retrieval-Augmented Generation (RAG) to ensure high-quality, relevant responses.

# Instructions:
Step 1: Open .env file and update the **MISTRAL_API_KEY** with your mistral api key.\
Step 2: Create python virtual environment and activate it.\
Step 3: Install all python packages using the command **pip install -r requirements.txt**.\
Step 4: Open Terminal and run the command **streamlit run app.py**.

# Notes: 
For users utilizing the free version of the Mistral API, the solution employs the PDF stored in the **data** folder to evaluate functionality. The following questions are designed to assist in verifying the solution's performance and capabilities.

Q1) What's cancer?
Q2) What are the symptoms of cancer?
Q3) Different types of cancer?
