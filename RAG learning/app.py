import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page Configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 Local RAG Chatbot with Ollama")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar for File Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file to specific temporary file structure
                    # We need to preserve extension for loaders to work sometimes or just be explicit
                    suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name

                    # Load Document
                    if suffix == ".pdf":
                        loader = PDFPlumberLoader(tmp_file_path)
                    else:
                        loader = TextLoader(tmp_file_path)
                    
                    documents = loader.load()

                    # Split Text
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        add_start_index=True
                    )
                    chunks = text_splitter.split_documents(documents)

                    # Embed and Store
                    # Using the same embedding model as in your notebooks
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    st.session_state.vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)

                    st.success("Document processed and embedded successfully!")
                    
                    # Cleanup temp file
                    os.unlink(tmp_file_path)

                except Exception as e:
                    st.error(f"Error processing document: {e}")

# Chat Interface
st.divider()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        if st.session_state.vector_store is None:
            response = "Please upload and process a document first."
            response_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            try:
                # Retriever
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                
                # LLM (Llama 3.1 as per your notebook)
                llm = ChatOllama(model="llama3.1", temperature=0)

                # Prompt Template
                template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
                prompt_template = ChatPromptTemplate.from_template(template)

                # Chain
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                # Stream response
                full_response = ""
                for chunk in chain.stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_msg = f"Error generating response: {e}"
                response_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
