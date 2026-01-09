from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from key import mykey
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Read PDF
reader = PdfReader("mypdf.pdf")
text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# 2. Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_text(text)

# 3. Create vector store
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_texts(chunks, embeddings)

# 4. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Prompt
prompt = ChatPromptTemplate.from_template(
    """
Answer the question using ONLY the context below.
If the answer is not found, say: "Not found in the document."

Context:
{context}

Question:
{question}
"""
)

# 6. LLM
llm = ChatOllama(
    model="phi3",
    temperature=0
)

# 7. RAG chain (LCEL – NO langchain.chains)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 8. Chat loop
while True:
    query = input("Ask your question (type 'stop' to exit): ")

    if query.lower().strip() == "stop":
        print("Chat ended.")
        break

    response = rag_chain.invoke(query)
    print("\nAnswer:\n", response.content, "\n")
