import os
from dotenv import load_dotenv, find_dotenv
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from bs4 import BeautifulSoup
from tqdm import tqdm

load_dotenv(find_dotenv())


def ingest_docs():
    """
    Ingests langchain documents to pinecone.
    """
    loader = ReadTheDocsLoader(path="./langchain-docs")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Splitted documents into {len(documents)} chunks.")

    # change the url from local to actual docs url
    for doc in tqdm(documents):
        # Get the file path from metadata
        file_path = doc.metadata["source"]
        # Ensure the file path is correct (you mentioned prepending "./")
        full_path = f"./{file_path}"

        # Read the HTML content from the file
        with open(full_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        # Parse the HTML to find the canonical link
        soup = BeautifulSoup(html_content, "html.parser")
        canonical_link_tag = soup.find("link", {"rel": "canonical"})

        if canonical_link_tag and canonical_link_tag.has_attr("href"):
            # Update the document metadata with the canonical link
            doc.metadata["source"] = canonical_link_tag["href"]

        # print(doc.metadata["source"])

    # Insert into pinecone
    print("Inserting documents into Pinecone...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    index_name = "langchain-doc-index"
    DIMENSIONS = 3072
    # vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    docsearch = PineconeVectorStore.from_documents(
        documents, embeddings, index_name=index_name
    )


if __name__ == "__main__":
    ingest_docs()
