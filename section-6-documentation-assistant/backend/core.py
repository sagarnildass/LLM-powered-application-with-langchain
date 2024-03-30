from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import Any, Dict, List

load_dotenv(find_dotenv())

index_name = "langchain-doc-index"

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
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    chat = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        condense_question_prompt=condense_question_prompt,
        memory=memory,
        get_chat_history=lambda h : h,
        return_source_documents=True
    )
    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    response = run_llm("What is RetrievalQA chain?")
    print(response)
