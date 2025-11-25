import pathlib, streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import glob
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# -----------------------------------

pdf_paths = glob.glob("data/Everstorm_*.pdf")
raw_docs = []

print(f"Loaded {len(raw_docs)} PDF pages from {len(pdf_paths)} files.")


for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    raw_docs.extend(loader.load())

chunks = []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(raw_docs)
print(f"✅ {len(chunks)} chunks ready for embedding")


embeddings = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
vectordb = FAISS.from_documents(chunks, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

vectordb.save_local("faiss_index")

print("✅ Vector store with", vectordb.index.ntotal, "embeddings")


# -----------------------------------


st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support Chatbot")

@st.cache_resource
def init_chain():
    vectordb = FAISS.load_local(
        "faiss_index",
        HuggingFaceEmbeddings(model_name="thenlper/gte-small"),
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    llm = Ollama(model="gemma3:1b", temperature=0.1)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
    )

chain = init_chain()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("What is in your mind?")
if question:
    with st.spinner("Thinking..."):
        response = chain(
            {
                "question": question,
                "chat_history": st.session_state.history,   # <- supply it
            }
        )
    st.session_state.history.append((question, response["answer"]))


for user, bot in reversed(st.session_state.history):
    st.markdown(f"**You:** {user}")
    st.markdown(bot)