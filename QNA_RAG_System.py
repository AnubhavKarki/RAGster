import streamlit as st
from streamlit_chat import message  # pip install streamlit-chat


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


"""# Chunking
Chunking is the process of breaking down large pieces of text into smaller segments. It's an essential technique that helps optimize the relevance of the content we get back from a vector database.

As a rule of thumb, if a chunk of text makes sense without the surrounding context to a human, it will make sense to the language model as well.

Finding the optimal chunk size for the documents in the corpus is crucial to ensure that the search results are accurate and relevant.
"""


def chunk_data(data, chunk_size=256, chunk_overlap=50):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks


"""# Calculate the cost for running embeddings"""


def print_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f"Total Tokens: {total_tokens}")
    print(f"Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}")


"""## Embedding and Uploading to a Vector Database (Pinecone)"""


# IMPORTANT: Ensure 'langchain-pinecone' is installed. You might need to run:
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


"""# Chroma

Chroma is an open source in-memory vector store, making fit for small to medium projects. It is an alternative to pinecone, but with less red tape.
"""


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


"""# Adding Memory (Chat History)"""

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


def main_rag_pipeline_gui():
    st.set_page_config(page_title="RAG Q&A", layout="wide")
    st.title("RAG Q&A Pipeline")

    # Sidebar for setup
    with st.sidebar:
        st.header("Setup")

        # Document upload
        uploaded_file = st.file_uploader("Upload PDF/Docx", type=["pdf", "docx"])
        wiki_query = st.text_input("Or Wikipedia query")

        # Vector store choice
        store_choice = st.selectbox("Vector Store", ["pinecone", "chroma"])

        if store_choice == "pinecone":
            index_name = st.text_input("Pinecone Index Name")
        else:
            persist_dir = st.text_input("Chroma Directory", value="/tmp/chroma_db")

        if st.button("Process Document"):
            with st.spinner("Processing..."):
                if uploaded_file:
                    # Save uploaded file
                    with open("temp_doc.pdf", "wb") as f:
                        f.write(uploaded_file.getvalue())
                    data = load_document("temp_doc.pdf")
                elif wiki_query:
                    data = load_from_wikipedia(wiki_query)

                chunks = chunk_data(data)
                st.success(f"Created {len(chunks)} chunks")

                print_embedding_cost(chunks)

                if store_choice == "pinecone":
                    vector_store = insert_or_fetch_embeddings(index_name, chunks)
                else:
                    vector_store = create_embeddings_chroma(chunks, persist_dir)

                st.session_state.vector_store = vector_store
                st.session_state.store_choice = store_choice
                st.rerun()

    # Chat interface (main area)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    for msg in st.session_state.messages:
        message(msg["content"], is_user=msg["role"] == "user")

    # Chat input
    if "vector_store" in st.session_state:
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

            with st.chat_message("user"):
                st.write(prompt)

            # Get answer using YOUR RAGWithMemory class!
            rag = RAGWithMemory(st.session_state.vector_store)
            answer = rag.invoke(prompt)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

            with st.chat_message("assistant"):
                st.write(answer)

    else:
        st.info("Upload a document and click 'Process Document' to start!")


if __name__ == "__main__":
    main_rag_pipeline_gui()
