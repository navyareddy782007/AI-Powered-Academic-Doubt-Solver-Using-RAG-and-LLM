import streamlit as st
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI-Powered Academic Doubt Solver",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Powered Academic Doubt Solver")
st.subheader("Using RAG and LLM")

st.write(
    "Upload your academic PDF and ask questions. "
    "The system retrieves relevant information from your "
    "study material and uses an LLM to generate the answer."
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# =========================================================
# CHROMADB CLIENT
# =========================================================

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(
        path="./chroma_db"
    )


client = get_chroma_client()


# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# SPLIT TEXT INTO CHUNKS
# =========================================================

def create_chunks(text, chunk_size=800, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def create_vector_database(chunks):

    try:
        client.delete_collection(
            name="academic_documents"
        )
    except Exception:
        pass

    collection = client.create_collection(
        name="academic_documents"
    )

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    return collection


# =========================================================
# RETRIEVE RELEVANT INFORMATION
# =========================================================

def retrieve_context(collection, question, top_k=4):

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    context = "\n\n".join(documents)

    return context


# =========================================================
# GENERATE ANSWER USING OLLAMA
# =========================================================

def generate_answer(question, context):

    prompt = f"""
You are an AI-powered academic doubt solver.

Your task is to answer the student's question using
ONLY the information provided in the Academic Context.

Rules:
1. Use only the retrieved academic context.
2. Do not invent or assume information.
3. If the answer is not available in the context, say:
   "I could not find the answer in the uploaded study material."
4. Explain the answer clearly and simply.
5. Give examples when the context contains examples.

Academic Context:
-------------------------
{context}
-------------------------

Student Question:
{question}

Answer:
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data["response"]

    except requests.exceptions.ConnectionError:

        return (
            "❌ Cannot connect to Ollama.\n\n"
            "Please make sure Ollama is running "
            "and the llama3.2 model is installed."
        )

    except requests.exceptions.Timeout:

        return (
            "❌ Ollama took too long to respond. "
            "Please try again."
        )

    except Exception as e:

        return f"❌ Error: {str(e)}"


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📚 Study Material")

uploaded_file = st.sidebar.file_uploader(
    "Upload your academic PDF",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    st.sidebar.success(
        f"Selected: {uploaded_file.name}"
    )

    if st.sidebar.button("📥 Process PDF"):

        with st.spinner("Processing your PDF..."):

            try:

                # Extract PDF text
                text = extract_text_from_pdf(
                    uploaded_file
                )

                if not text.strip():

                    st.error(
                        "❌ No readable text was found "
                        "in this PDF."
                    )

                else:

                    # Create chunks
                    chunks = create_chunks(text)

                    # Create vector database
                    collection = create_vector_database(
                        chunks
                    )

                    # Store collection in session state
                    st.session_state.collection = collection
                    st.session_state.pdf_processed = True
                    st.session_state.document_name = (
                        uploaded_file.name
                    )

                    st.success(
                        f"✅ PDF processed successfully!\n\n"
                        f"Created {len(chunks)} text chunks."
                    )

            except Exception as e:

                st.error(
                    f"❌ Error processing PDF: {str(e)}"
                )


# =========================================================
# MAIN QUESTION SECTION
# =========================================================

st.divider()

st.header("💡 Ask Your Academic Doubt")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is normalization in DBMS?"
)


# =========================================================
# ASK QUESTION
# =========================================================

if st.button("🔍 Ask Question"):

    if uploaded_file is None:

        st.warning(
            "⚠️ Please upload an academic PDF first."
        )

    elif not st.session_state.get(
        "pdf_processed",
        False
    ):

        st.warning(
            "⚠️ Please click 'Process PDF' first."
        )

    elif not question.strip():

        st.warning(
            "⚠️ Please enter your question."
        )

    else:

        with st.spinner(
            "🔎 Searching study material..."
        ):

            try:

                collection = (
                    st.session_state.collection
                )

                context = retrieve_context(
                    collection,
                    question,
                    top_k=4
                )

            except Exception as e:

                st.error(
                    f"❌ Retrieval error: {str(e)}"
                )

                context = ""

        if context:

            with st.spinner(
                "🤖 Generating answer using LLM..."
            ):

                answer = generate_answer(
                    question,
                    context
                )

            st.subheader("🤖 Answer")

            st.write(answer)

            with st.expander(
                "📖 View Retrieved Study Material"
            ):

                st.write(context)


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.subheader("🔧 Technologies Used")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("📄 **PDF Reader**")
    st.write("PyPDF")

with col2:
    st.write("🧠 **Embeddings**")
    st.write("Sentence Transformers")

with col3:
    st.write("🗄️ **Vector Database**")
    st.write("ChromaDB")

with col4:
    st.write("🤖 **LLM**")
    st.write("Ollama + Llama 3.2")


st.caption(
    "AI-Powered Academic Doubt Solver | RAG + LLM"
)