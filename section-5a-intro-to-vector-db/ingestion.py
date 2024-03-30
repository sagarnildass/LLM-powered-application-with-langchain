from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    loader = TextLoader("./mediumblogs/mediumblog1.txt")
    document = loader.load()
    # print(document)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    index_name = "udemy-langchain"
    DIMENSIONS = 3072
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    docsearch = PineconeVectorStore.from_documents(
        texts, embeddings, index_name=index_name
    )

    # qa = RetrievalQA.from_chain_type(llm=OpenAI(),
    #                                 chain_type="stuff",
    #                                 retriever=docsearch.as_retriever(),
    #                                 return_source_documents=True)
    # query = "What is a vector DB? Give me a 15 word answer for a beginner"
    # result = qa.invoke({"query": query})
    # response = result["result"]
    # print(response)
