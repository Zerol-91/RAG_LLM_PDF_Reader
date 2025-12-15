import streamlit as st
from openai import OpenAI
from pypdf import PdfReader # Библиотека для чтения PDF
import chromadb 
import os
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer 
import sqlite3 
import datetime

# --- НАСТРОЙКИ ---
st.set_page_config(page_title="RAG + DataBase", page_icon="📄")
st.title("Чат с Памятью (SQLite + LLM)")

load_dotenv() 
api_key = os.getenv("OPENROUTER_API_KEY")


if not api_key:
    st.error("Не найден ключ API! Создайте файл .env и впишите туда OPENROUTER_API_KEY")
    st.stop()

# OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

@st.cache_resource# Декоратор для единоразовой загрузки MiniLM
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()


chroma_client = chromadb.PersistentClient(path="my_vector_db")
collection = chroma_client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"} 
)

DB_FOLDER = "data"
DB_FILE = os.path.join(DB_FOLDER, "chat_history.db")

os.makedirs(DB_FOLDER, exist_ok=True)

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()


# Init table
def init_db():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

def save_message_to_db(session_id, role, content):
    cursor.execute(
        'INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
        (session_id, role, content)
    )
    conn.commit()

    
def load_history_from_db(session_id, limit=20):
    cursor.execute(
        'SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?',
        (session_id, limit)
    )
    rows = cursor.fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows][::-1]

# start table creation at startup
init_db() 


# Rag Functions
def get_pdf_text(uploaded_file):
    text = ""
    try:
        pdf_reader = PdfReader(uploaded_file)
        # Читаем каждую страницу
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Ошибка чтения PDF: {e}")
    return text


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 50: # Игнорируем совсем мелкие кусочки
            chunks.append(chunk)
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        model="all-minilm", 
        input=text
    )
    return response.data[0].embedding


def get_embedding(text):
    return embedding_model.encode(text).tolist()

def get_existing_files():
    data = collection.get(include=['metadatas'])
    
    unique_files = set([item['source'] for item in data['metadatas']])
    return list(unique_files)


# Interface
CURRENT_SESSION_ID = "user_default"
if "messages" not in st.session_state:
    db_history = load_history_from_db(CURRENT_SESSION_ID)
    st.session_state.messages = db_history

    if not st.session_state.messages:
        st.session_state.messages = []
        welcome_msg = "Привет! Я твой RAG-помощник. Загрузи PDF или просто задай вопрос."
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})



with st.sidebar:
    st.header("Загрузка")
    uploaded_file = st.file_uploader("Выберите PDF файл", type="pdf")
    
    if uploaded_file:
        filename = uploaded_file.name
        existing_docs = collection.get(where={"source": filename})
        
        if len(existing_docs['ids']) > 0:
            st.success(f"Файл '{filename}' уже есть в базе.")
        else:
            with st.spinner("Индексирую новый файл..."):
                text = get_pdf_text(uploaded_file)
                chunks = split_text(text)
                

                ids = []       
                metadatas = [] 
                vectors = []   
                documents_text = [] 
                
                progress = st.progress(0)
                for i, chunk in enumerate(chunks):
                    vec = get_embedding(chunk)
                    
                    ids.append(f"{filename}_chunk{i}")
                    metadatas.append({"source": filename})
                    vectors.append(vec)
                    documents_text.append(chunk)
                    
                    progress.progress((i+1)/len(chunks))

                metadatas = [{"source": filename} for _ in chunks]

                collection.add(
                    ids=ids,
                    embeddings=vectors,
                    documents=documents_text, 
                    metadatas=metadatas
                )
                st.success("Сохранено в базу.")

    if st.button("🗑️ Очистить историю чата"):
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (CURRENT_SESSION_ID,))
        conn.commit()
        st.session_state.messages = []
        st.rerun() # Перезагрузить страницу

    st.divider()
    files_list = get_existing_files()
    options = ["Во всей базе"] + files_list
    selected_file = st.selectbox("Где искать ответ?", options)

# Chat drawing
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Вопрос..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    save_message_to_db(CURRENT_SESSION_ID, "user", prompt)
    
    query_vec = get_embedding(prompt)
    search_params = {
        "query_embeddings": [query_vec],
        "n_results": 10
    }
    
    if selected_file != "Во всей базе":
        search_params["where"] = {"source": selected_file}

    results = collection.query(**search_params)
    valid_chunks = []

    # Information found in the database
    with st.expander("Техническая информация (Что нашла база)"):
        found_chunks = results['documents'][0]
        distances = results['distances'][0]
            
        for i, dist in enumerate(distances):
            chunk_text = found_chunks[i]
            st.write(f"**Кусок {i+1}** (Дистанция: {dist:.4f}):")
            st.caption(chunk_text[:200] + "...") # Показываем начало куска
                
            # Фильтр: берем только если дистанция меньше 0.7 (можно менять)
            if dist < 0.7:
                st.success("Подходит")
                valid_chunks.append(chunk_text)
            else:
                st.warning("Этот кусок отброшен (слишком непохож)")

 
    if not valid_chunks:
        system_prompt = "Ты умный и полезный ассистент."
    else:
        context_text = "\n---\n".join(valid_chunks)
        system_prompt = f"""
        Ты — умный помощник. Пользователь загрузил документы, и ниже приведено их содержимое.
        Твоя задача — отвечать на вопросы пользователя ТОЛЬКО на основе этого содержимого.
        Не говори "я не вижу файлов" или "в предоставленном тексте". Отвечай так, будто ты прочитал этот документ целиком.

        Содержимое документа:\n{context_text}
        """

    # Генерация (OpenRouter)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct:free", # Или "google/gemma-2-9b-it:free"
                messages=[
                    {"role": "system", "content": system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
                ],
                stream=True,
                extra_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Local RAG App"
                }
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌") # ▌ - это курсор
            message_placeholder.markdown(full_response) # Финальный текст без курсора
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Save in Database
            save_message_to_db(CURRENT_SESSION_ID, "assistant", full_response)

        except Exception as e:
            st.error(f"Ошибка API: {e}")
        
