from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore

load_dotenv(find_dotenv())

index_name = "langchain-doc-index"

def run_llm(query:str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    chat = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    return qa.invoke({"query": query})

if __name__ == "__main__":
    response = run_llm("What is RetrievalQA chain?")
    print(response)
