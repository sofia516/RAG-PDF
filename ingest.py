from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "data/research_paper.pdf"
VECTORSTORE_PATH = "vectorstore"


# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages")


# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# 3. Create LOCAL embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating embeddings...")


# 4. Store embeddings in FAISS
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


# 5. Save FAISS database
vectorstore.save_local(VECTORSTORE_PATH)

print("FAISS vector store created successfully!")