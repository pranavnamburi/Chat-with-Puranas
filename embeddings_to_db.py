from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer 


pdf_loader = PyPDFDirectoryLoader("./pdf-docs" )

loaders = [pdf_loader]

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)

print(f"Total number of documents: {len(all_documents)}")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class Embedder:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, documents):
        texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        embeddings = [self.model.encode(text, show_progress_bar=True).tolist() for text in texts]
        return embeddings
embedder = Embedder(model)


db = Chroma(embedding_function= embedder, persist_directory="./chromadb")
retv = db.as_retriever()

batch_size = 100
for i in range(0, len(all_documents), batch_size):
    batch_documents = all_documents[i:i + batch_size]
    batch_embeddings = embedder.embed_documents(batch_documents)
    retv.add_documents(batch_documents, embeddings=batch_embeddings)
    print(f"Processed {i + len(batch_documents)} documents")

       
db.persist()