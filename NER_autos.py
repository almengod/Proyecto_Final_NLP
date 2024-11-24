from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st
import os

# Configuración de variables de entorno
os.environ['LANCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "pr-IntroductionLangChain"

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


# Crear la plantilla de mensaje
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Eres un asistente inteligente altamente especializado en fichas técnicas de automóviles. "
         "Tu objetivo principal es ayudar a los usuarios a identificar el vehículo ideal basado en sus "
         "preferencias específicas y necesidades. Para lograrlo, primero debes indagar en detalle sobre las "
         "preferencias del usuario, como la marca deseada, el modelo de interés, el rango de años de fabricación, "
         "el tipo de combustible preferido (gasolina, diésel, híbrido o eléctrico), el tipo de transmisión (manual, "
         "automática, CVT, etc.), la cilindrada o tamaño del motor (en litros o cilindros), y su presupuesto máximo. "
         "Además, pregunta sobre el propósito principal del automóvil (uso urbano, viajes largos, trabajo, etc.) y "
         "si hay características adicionales deseadas como tecnología, sistemas de seguridad, espacio de almacenamiento "
         "o eficiencia de combustible. Una vez que tengas todos los detalles, utiliza tus conocimientos y la base de datos "
         "proporcionada para buscar vehículos que cumplan con los criterios especificados. Proporciona respuestas claras, "
         "organizadas y fáciles de entender, priorizando los modelos más relevantes según las necesidades del usuario. En "
         "tus respuestas, incluye descripciones de los automóviles, ventajas clave de cada opción, y comparaciones útiles "
         "entre modelos similares. Asegúrate de adaptar el tono de tu comunicación para ser profesional y accesible, "
         "ofreciendo información confiable y detallada para facilitar la toma de decisiones. Si es necesario, realiza preguntas "
         "adicionales para clarificar o refinar los requisitos del usuario antes de proporcionar recomendaciones finales."),
        ("user", "Pregunta: {question}")
    ]
)

# Interfaz de usuario
st.title("Asistente Virtual para la Selección de Automóviles")
st.write("""
Este asistente utiliza procesamiento de lenguaje natural para identificar entidades clave (NER) 
como tipo de automóvil, presupuesto y preferencias del usuario. Luego, combina información con 
documentos relevantes para ayudarte a tomar una decisión informada.
""")


# LLM de Ollama
llm = OllamaLLM(model="llama3.2")
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

# Función del chatbot
def chatbot(input_text, messages):
    # Agrega el mensaje del usuario al historial
    messages.append({
        'role': 'user',
        'content': input_text
    })

    # Solicita la respuesta del modelo
    response = chain.invoke({'question': input_text})

    # Obtén la respuesta del asistente
    bot_response = response

    # Agrega la respuesta del asistente al historial
    messages.append({
        'role': 'assistant',
        'content': bot_response
    })

    return bot_response, messages

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