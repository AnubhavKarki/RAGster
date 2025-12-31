# RAG Q&A System - Modular Pipeline

def load_document(file):
    import os

    name, ext = os.path.splitext(file)

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        print(f"loading {file}")
        loader = PyPDFLoader(file)

    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        print(f"loading {file}")
        loader = Docx2txtLoader(file)

    else:
        print("Document format is not supported!")

    data = loader.load()
    return data


# Wikipedia
def load_from_wikipedia(query, lang="en", load_max_docs=1):
    from langchain_community.document_loaders import WikipediaLoader

    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data



def chunk_data(data, chunk_size=256, chunk_overlap=50):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks


# Calculate the cost for running embeddings
def print_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f"Total Tokens: {total_tokens}")
    print(f"Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}")


# Embedding and Uploading to a Vector Database (Pinecone)
def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import ServerlessSpec
    from langchain_pinecone import (
        Pinecone as PineconeLangChain,
    )  # Use the updated Pinecone integration

    # Pinecone client
    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

    # Check if the index already exists using the Pinecone client v3.x API
    # pc.list_indexes() returns a list of IndexModel objects
    if index_name in [idx.name for idx in pc.list_indexes()]:
        print(f"Index {index_name} already exists. Loading embeddings ... ", end="")
        vector_store = PineconeLangChain.from_existing_index(index_name, embeddings)
        print("Ok")
    else:
        print(f"Creating index {index_name} and embeddings ...", end="")
        # Create a new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        vector_store = PineconeLangChain.from_documents(
            chunks, embeddings, index_name=index_name
        )
        print("Ok")
    return vector_store


def delete_pinecone_index(index_name="all"):
    import pinecone

    pc = pinecone.Pinecone()

    if index_name == "all":
        indexes = pc.list_indexes().names()
        print("Deleting all indexes ... ")
        for index in indexes:
            pc.delete_index(index)
        print("Ok")
    else:
        print(f"Deleting index {index_name} ...", end="")
        pc.delete_index(index_name)
        print("Ok")


def ask_and_get_answer(vector_store, q, k=3):
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from operator import itemgetter  # Import itemgetter for extracting values
    from langchain_core.runnables import (
        RunnableParallel,
        RunnablePassthrough,
    )  # Import RunnableParallel

    llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=1)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    prompt = ChatPromptTemplate.from_template(
        "Context: {context}\n\nQuestion: {input}\n\nAnswer:"
    )

    # Correctly structure the chain to pass only the query string to the retriever
    setup_and_retrieval = RunnableParallel(
        {
            "context": itemgetter("input")
            | retriever,  # Pass only the 'input' string to the retriever
            "input": itemgetter(
                "input"
            ),  # Pass only the 'input' string to the next step's input slot
        }
    )

    chain = setup_and_retrieval | prompt | llm

    return chain.invoke({"input": q}).content

# Chroma
# Create Embeddings:
def create_embeddings_chroma(chunks, persist_directory="./"):
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

    vector_store = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_directory
    )
    return vector_store


def load_embeddings_chroma(persist_directory="./chroma_db"):
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    vector_store = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )

    return vector_store


# Adding Memory (Chat History)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter


class RAGWithMemory:
    def __init__(self, vector_store, k=3, model="gpt-4o-mini"):
        """RAG chain with conversation memory."""
        self.llm = ChatOpenAI(model=model, temperature=1)
        self.retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        self.message_history = ChatMessageHistory()

        # Single source of truth prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant. Use the conversation history and provided context to answer questions.
Answer based only on the context. If you don't know, say so.""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Context: {context}\n\nQuestion: {input}\n\nAnswer:"),
            ]
        )

        self.setup_and_retrieval = RunnableParallel(
            {
                "context": itemgetter("input") | self.retriever,
                "input": itemgetter("input"),
                "history": itemgetter("history"),
            }
        )

        self.chain = self.setup_and_retrieval | self.prompt | self.llm

    def invoke(self, question):
        """Get answer with automatic history management."""
        result = self.chain.invoke(
            {"input": question, "history": self.message_history.messages}
        )

        answer = result.content
        self.message_history.add_user_message(question)
        self.message_history.add_ai_message(answer)
        return answer

    def clear_history(self):
        """Clear conversation history."""
        self.message_history.clear()

    def get_history(self):
        """Get current conversation history."""
        return self.message_history.messages


def main_rag_pipeline():
    """Complete modular RAG pipeline for PDF/Docx/Wikipedia with Pinecone & Chroma."""
    print("\n" + "=" * 70)
    print("           RAG Q&A Pipeline")
    print("=" * 70 + "\n")

    vector_store = None
    store_choice = None
    restart_pipeline = True

    while restart_pipeline:
        try:
            if vector_store is None:
                # 1. Load document
                print("1. Load Document:")
                print("-" * 30)
                file_input = input(
                    "Enter filename (.pdf, .docx) or 'wikipedia' for Wikipedia: "
                ).strip()

                if file_input.lower() == "help":
                    print_help_menu()
                    continue

                if not file_input:
                    print("Invalid input. Please try again.")
                    continue

                if file_input.lower() == "wikipedia":
                    query = input("Enter Wikipedia query: ").strip()
                    if not query:
                        print("Invalid query. Please try again.")
                        continue
                    data = load_from_wikipedia(query, load_max_docs=2)
                    print(f"Loaded {len(data)} Wikipedia pages\n")
                else:
                    data = load_document(file_input)
                    print(f"Loaded {len(data)} pages from {file_input}\n")

                # 2. Chunk data
                print("2. Chunking data...")
                print("-" * 30)
                chunks = chunk_data(data)
                print(f"Created {len(chunks)} chunks\n")

                # 3. Show embedding cost
                print("3. Embedding Cost:")
                print("-" * 30)
                print_embedding_cost(chunks)
                print()

                # 4. Choose vector store & create embeddings
                print("4. Choose Vector Store:")
                print("-" * 30)
                store_choice = input("Enter 'pinecone' or 'chroma': ").strip().lower()

                if store_choice.lower() == "help":
                    print_help_menu()
                    continue

                if store_choice == "pinecone":
                    index_name = input("Enter Pinecone index name: ").strip()
                    if not index_name:
                        print("Invalid index name. Please try again.")
                        continue
                    print("\nCreating/Loading Pinecone embeddings...")
                    vector_store = insert_or_fetch_embeddings(index_name, chunks)
                    print("Pinecone ready!")
                    print()

                elif store_choice == "chroma":
                    persist_dir = input(
                        "Enter Chroma persist directory (press Enter for /tmp/chroma_db): "
                    ).strip()
                    if not persist_dir:
                        persist_dir = "/tmp/chroma_db"
                    else:
                        if persist_dir.startswith("./"):
                            persist_dir = f"/tmp/{persist_dir.lstrip('./')}"

                    import os

                    os.makedirs(persist_dir, exist_ok=True)

                    print(f"\nCreating Chroma embeddings in {persist_dir}...")
                    vector_store = create_embeddings_chroma(chunks, persist_dir)
                    print("Chroma ready!")
                    print()
                else:
                    print("Invalid choice. Please enter 'pinecone' or 'chroma'.")
                    continue

            # 5. Q&A Loop with Memory (CLASS VERSION)
            print("=" * 70)
            print("Q&A Ready!")
            print("Commands: 'quit', 'clear', 'delete_index', 'history', 'help'")
            print("=" * 70 + "\n")

            rag = RAGWithMemory(vector_store)

            while True:
                try:
                    print(f"Messages in history: {len(rag.message_history.messages)}")
                    print("-" * 50)

                    question = input("Your question: ").strip()
                    print()

                    if question.lower() in ["quit", "exit", "q"]:
                        print("Goodbye!")
                        return

                    elif question.lower() == "clear":
                        rag.clear_history()
                        print("History cleared!")
                        print()
                        continue

                    elif question.lower() == "delete_index":
                        confirm = input("Delete indexes? (yes/no): ").strip().lower()
                        if confirm in ["yes", "y"]:
                            try:
                                if store_choice == "pinecone":
                                    delete_pinecone_index("all")
                                elif store_choice == "chroma":
                                    import shutil, os

                                    persist_dir = input(
                                        "Enter Chroma directory to delete: "
                                    ).strip()
                                    if persist_dir.startswith("./"):
                                        persist_dir = f"/tmp/{persist_dir.lstrip('./')}"
                                    shutil.rmtree(persist_dir, ignore_errors=True)
                                    print("Chroma directory deleted!")
                                print("Indexes deleted!")

                                # Ask if user wants to restart pipeline
                                restart = (
                                    input(
                                        "Start new pipeline with new document/index? (yes/no): "
                                    )
                                    .strip()
                                    .lower()
                                )
                                if restart in ["yes", "y"]:
                                    vector_store = None
                                    store_choice = None
                                    restart_pipeline = True
                                    break  # Break Q&A loop to restart pipeline
                                else:
                                    print("Returning to Q&A...")
                            except Exception as e:
                                print(f"Error deleting indexes: {str(e)}")
                        print()
                        continue

                    elif question.lower() == "history":
                        print("\nChat History:")
                        print("-" * 40)
                        recent_history = rag.get_history()[-6:]
                        if not recent_history:
                            print("No history yet.")
                        else:
                            for i, msg in enumerate(recent_history, 1):
                                from langchain_core.messages import HumanMessage

                                role = (
                                    "User"
                                    if isinstance(msg, HumanMessage)
                                    else "Assistant"
                                )
                                print(f"{i}. {role}: {msg.content}")
                        print("-" * 40)
                        print()
                        continue

                    elif question.lower() == "help":
                        print_help_menu()
                        print()
                        continue

                    # Get answer with history
                    answer = rag.invoke(question)

                    print("Answer:")
                    print("-" * 40)
                    print(answer)
                    print("-" * 40)
                    print()

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    return
                except Exception as e:
                    print(f"Error processing question: {str(e)}")
                    print("Please try again.\n")
                    continue

        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again.\n")
            continue


def print_help_menu():
    """Print help menu with all commands."""
    print("\nAvailable Commands:")
    print("-" * 30)
    print("'quit', 'exit', 'q'     - Exit the application")
    print("'clear'                 - Clear conversation history")
    print("'history'               - Show recent chat history")
    print("'delete_index'          - Delete current indexes")
    print("'help'                  - Show this help menu")
    print("'pinecone' or 'chroma'  - Vector store choices")
    print("'wikipedia'             - Load Wikipedia data")
    print("-" * 30 + "\n")


# def main_rag_pipeline_gui():
#     st.set_page_config(page_title="RAG Q&A", layout="wide")
#     st.title("RAG Q&A Pipeline")

#     # Sidebar for setup
#     with st.sidebar:
#         st.header("Setup")

#         # Document upload
#         uploaded_file = st.file_uploader("Upload PDF/Docx", type=["pdf", "docx"])
#         wiki_query = st.text_input("Or Wikipedia query")

#         # Vector store choice
#         store_choice = st.selectbox("Vector Store", ["pinecone", "chroma"])

#         if store_choice == "pinecone":
#             index_name = st.text_input("Pinecone Index Name")
#         else:
#             persist_dir = st.text_input("Chroma Directory", value="/tmp/chroma_db")

#         if st.button("Process Document"):
#             with st.spinner("Processing..."):
#                 if uploaded_file:
#                     # Save uploaded file
#                     with open("temp_doc.pdf", "wb") as f:
#                         f.write(uploaded_file.getvalue())
#                     data = load_document("temp_doc.pdf")
#                 elif wiki_query:
#                     data = load_from_wikipedia(wiki_query)

#                 chunks = chunk_data(data)
#                 st.success(f"Created {len(chunks)} chunks")

#                 print_embedding_cost(chunks)

#                 if store_choice == "pinecone":
#                     vector_store = insert_or_fetch_embeddings(index_name, chunks)
#                 else:
#                     vector_store = create_embeddings_chroma(chunks, persist_dir)

#                 st.session_state.vector_store = vector_store
#                 st.session_state.store_choice = store_choice
#                 st.rerun()

#     # Chat interface (main area)
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Chat history
#     for msg in st.session_state.messages:
#         message(msg["content"], is_user=msg["role"] == "user")

#     # Chat input
#     if "vector_store" in st.session_state:
#         if prompt := st.chat_input("Ask a question..."):
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             st.rerun()

#             with st.chat_message("user"):
#                 st.write(prompt)

#             # Get answer using YOUR RAGWithMemory class!
#             rag = RAGWithMemory(st.session_state.vector_store)
#             answer = rag.invoke(prompt)

#             st.session_state.messages.append({"role": "assistant", "content": answer})
#             st.rerun()

#             with st.chat_message("assistant"):
#                 st.write(answer)

#     else:
#         st.info("Upload a document and click 'Process Document' to start!")

import streamlit as st
from streamlit_chat import message
import time

def main_rag_pipeline_gui():
    # Modern config
    st.set_page_config(
        page_title="RAGister AI", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for modern look
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 1.5rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    .user-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        margin-left: auto;
        color: white;
    }
    .assistant-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .status-bar {
        background: rgba(0,0,0,0.05);
        padding: 1rem;
        border-radius: 1rem;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">RAGister AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Minimal functional sidebar - ALL UNIQUE KEYS
    with st.sidebar:
        st.header("Setup")
        
        # Radio with UNIQUE KEY
        doc_type = st.radio(
            "Document Source", 
            ["Upload File", "Wikipedia"],
            key="doc_source_radio_unique"
        )
        
        uploaded_file = None
        wiki_query = ""
        
        if doc_type == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose PDF/Docx", 
                type=["pdf", "docx"],
                key="file_uploader_key_unique"
            )
        else:
            wiki_query = st.text_input(
                "Wikipedia Query", 
                key="wiki_query_input_unique"
            )
        
        # Vector store choice with UNIQUE KEY
        store_choice = st.selectbox(
            "Vector Store", 
            ["pinecone", "chroma"], 
            key="store_choice_select_unique"
        )
        
        # Store-specific inputs with UNIQUE KEYS
        if store_choice == "pinecone":
            index_name = st.text_input(
                "Pinecone Index", 
                key="pinecone_index_input_unique"
            )
        else:
            persist_dir = st.text_input(
                "Chroma Directory", 
                value="/tmp/chroma_db", 
                key="chroma_dir_input_unique"
            )
        
        # Process button with UNIQUE KEY
        process_btn = st.button(
            "Process Document", 
            type="primary", 
            key="process_button_unique"
        )
    
    # Status notifications
    if "status" in st.session_state:
        with st.container():
            status_class = "status-bar"
            st.markdown(f"""
            <div class="{status_class}">
                <strong>{st.session_state.status['title']}:</strong> 
                {st.session_state.status['message']}
            </div>
            """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "embedding_cost" not in st.session_state:
        st.session_state.embedding_cost = 0
    
    # Process document
    if process_btn and (uploaded_file or wiki_query):
        with st.spinner("Processing document..."):
            try:
                # Load document
                if uploaded_file:
                    suffix = uploaded_file.name.split('.')[-1].lower()
                    filename = f"temp_doc.{suffix}"
                    
                    with open(filename, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    data = load_document(filename)
                    
                    # Cleanup
                    import os
                    if os.path.exists(filename):
                        os.remove(filename)
                    
                    st.session_state.status = {
                        "title": "Document Loaded",
                        "message": f"{len(data)} pages from {uploaded_file.name}",
                        "success": True
                    }
                    
                elif wiki_query:
                    data = load_from_wikipedia(wiki_query, load_max_docs=2)
                    st.session_state.status = {
                        "title": "Wikipedia Loaded", 
                        "message": f"{len(data)} pages on '{wiki_query}'",
                        "success": True
                    }
                
                # Chunking
                chunks = chunk_data(data)
                st.session_state.status = {
                    "title": "Chunking Complete",
                    "message": f"Created {len(chunks)} chunks",
                    "success": True
                }
                
                # Embedding cost
                import tiktoken
                enc = tiktoken.encoding_for_model('text-embedding-3-small')
                total_tokens = sum([len(enc.encode(page.page_content)) for page in chunks])
                cost_usd = total_tokens / 1000 * 0.0004
                st.session_state.embedding_cost = cost_usd
                
                st.session_state.status = {
                    "title": "Embedding Cost",
                    "message": f"{total_tokens:,} tokens = ${cost_usd:.6f}",
                    "success": True
                }
                
                # Create vector store
                if store_choice == "pinecone" and index_name:
                    vector_store = insert_or_fetch_embeddings(index_name, chunks)
                    st.session_state.status = {
                        "title": "Pinecone Ready",
                        "message": f"Index '{index_name}' created/loaded",
                        "success": True
                    }
                elif store_choice == "chroma":
                    import os
                    os.makedirs(persist_dir, exist_ok=True)
                    vector_store = create_embeddings_chroma(chunks, persist_dir)
                    st.session_state.status = {
                        "title": "Chroma Ready", 
                        "message": f"Database at {persist_dir}",
                        "success": True
                    }
                
                st.session_state.vector_store = vector_store
                st.session_state.store_choice = store_choice
                st.rerun()
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.session_state.status = {
                    "title": "Error", 
                    "message": str(e),
                    "success": False
                }
    
    # Chat interface (main area)
    chat_container = st.container()
    
    with chat_container:
        # Show chat history
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input (only if vector store ready)
        if st.session_state.vector_store:
            if prompt := st.chat_input("Ask a question about your document..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        try:
                            rag = RAGWithMemory(st.session_state.vector_store)
                            answer = rag.invoke(prompt)
                            
                            st.markdown(answer)
                            st.session_state.status = {
                                "title": "Response Generated",
                                "message": "Answer ready!",
                                "success": True
                            }
                            
                            # Add to history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer
                            })
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Response failed: {str(e)}")
                            st.session_state.status = {
                                "title": "Response Error",
                                "message": str(e),
                                "success": False
                            }
        else:
            # Welcome screen
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                ### Get Started
                1. **Upload** PDF/Docx OR enter Wikipedia topic
                2. **Configure** vector store (Pinecone/Chroma)
                3. **Click** "Process Document" 
                4. **Start asking** questions!
                
                *Cost shown before processing - No surprises.*
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("*RAGister - Production RAG Pipeline*")

if __name__ == "__main__":
    main_rag_pipeline_gui()