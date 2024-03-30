from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import os
from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


from pinecone import Pinecone

load_dotenv(find_dotenv())

# pc = Pinecone(api_key=os.getenv["PINECONE_API_KEY"])

INDEX_NAME = "udemy-langchain"

# Define your system instruction
system_instruction = "The assistant should provide detailed explanations."

# Define your template with the system instruction
template = (
    f"{system_instruction} "
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)

# Create the prompt template
condense_question_prompt = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False, output_key='answer')


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        condense_question_prompt=condense_question_prompt,
        memory=memory,
        get_chat_history=lambda h : h
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
