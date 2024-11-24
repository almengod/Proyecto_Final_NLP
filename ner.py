import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os

# Configuración inicial
load_dotenv()
os.environ['LANCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "AutoNERProject"

# Función para cargar y procesar documentos
@st.cache_data
def cargar_y_crear_faiss(pdf_paths):
    documentos = []
    for pdf in pdf_paths:
        documentos.extend(PyPDFLoader(pdf).load())
    embeddings = OllamaEmbeddings(model="llama3.2")
    faiss = FAISS.from_documents(documentos, embeddings)
    faiss.save_local("faiss_index")
    return faiss

# Carga o creación de FAISS
pdf_paths = [
    'data/Descripción de vehiculo.pdf',
    'data/El automovil.pdf',
    'data/Redalyc.Criterios técnicos para evaluar y seleccionar ofertas de vehículos ligeros_.pdf'
]
try:
    faiss = FAISS.load_local("faiss_index", OllamaEmbeddings(model="llama3.2"))
except:
    faiss = cargar_y_crear_faiss(pdf_paths)

retriever = faiss.as_retriever()

# Configuración del modelo
llm = OllamaLLM(model="llama3.2")

# Prompt para las tareas de NER
ner_prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "Eres un experto en la selección de automóviles. Identifica entidades clave como el tipo de automóvil, la categoría, el presupuesto, y cualquier preferencia del usuario para recomendar opciones óptimas."},
    {"role": "user", "content": "{question}"}
])

# Cadena de RAG
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, prompt=ner_prompt)

# Función del chatbot con RAG
def chatbot(input_text, messages):
    # Agregar el mensaje del usuario al historial
    messages.append(('user', input_text))
    
    # Reducir el historial de mensajes si es demasiado largo
    MAX_HISTORY_LENGTH = 10  # Ajusta según tus necesidades
    if len(messages) > MAX_HISTORY_LENGTH:
        messages = messages[-MAX_HISTORY_LENGTH:]
    
    # Crear el diccionario de entrada
    inputs = {
        'question': input_text,
        'chat_history': [(m[1] if m[0] == "user" else m[1]) for m in messages]
    }
    
    # Llamar al chain con el historial
    response = chain(inputs)
    
    # Agregar la respuesta del asistente al historial
    messages.append(('assistant', response['answer']))
    
    return response['answer'], messages

# Interfaz de usuario
st.title("Asistente Virtual para la Selección de Automóviles")
st.write("""
Este asistente utiliza procesamiento de lenguaje natural para identificar entidades clave (NER) 
como tipo de automóvil, presupuesto y preferencias del usuario. Luego, combina información con 
documentos relevantes para ayudarte a tomar una decisión informada.
""")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)

# Entrada del usuario
user_input = st.text_input("Escribe tu pregunta (ej. 'Quiero un sedán económico para viajes largos'):")

if user_input:
    response, st.session_state.messages = chatbot(user_input, st.session_state.messages)
    st.chat_message("assistant").markdown(response)

# Botón para reiniciar conversación
if st.button("Reiniciar conversación"):
    st.session_state.messages = []
    st.experimental_rerun()
