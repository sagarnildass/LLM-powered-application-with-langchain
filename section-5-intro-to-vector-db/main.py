from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import os
from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone

load_dotenv(find_dotenv())
 
# pc = Pinecone(api_key=os.getenv["PINECONE_API_KEY"])
 
INDEX_NAME = "udemy-langchain"
 
 
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
 
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever()
    )
    return qa.invoke({"question": query, "chat_history": chat_history})

if __name__ == "__main__":
    chat_history = []
    while True:
        query = input("You: ")
        response = run_llm(query, chat_history)

        # Assuming the actual answer is under response['answer'], adjust if necessary
        # actual_answer = response.get('answer') if isinstance(response, dict) else response
        chat_history.append((query, response["answer"]))
        
        print("Bot: {}".format(response["answer"]))
