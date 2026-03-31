from pypdf import PdfReader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Config Data
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_file_comtent(file_path) -> list[Document]:
    reader = PdfReader(file_path)
    documents = []
    filename = os.path.basename(file_path)
    for page_num, page in enumerate(reader.pages):
        documents.append(Document(
            page_content=page.extract_text(),
            metadata={"source": filename, "page": page_num + 1}
        ))
    return documents


def chunk_content(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def create_vector_store(chunks: list[Document]):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
        )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def maim()->None:
    print(f"="*20 + " PaperMindAgent" + "="*20)
    print("Indexing papers ...")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    all_chunks = []
    for pdf_file in pdf_files:
        print(f"reading and indexing: {pdf_file}")
        file_chunks = extract_file_comtent(os.path.join(data_dir, pdf_file))
        chunks = chunk_content(file_chunks)
        all_chunks.extend(chunks)
    vector_store = create_vector_store(all_chunks)
    vector_store.save_local("vector_store.index")
    print("Indexing completed and saved to vector_store.index.")
    print(f"="*50)
    print("You can now run the agent to query.py to query the indexed papers.")
        
if __name__ == "__main__":
    maim()