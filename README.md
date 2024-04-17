# TALLER 9: Large Language Model

Este proyecto implementa un sistema de búsqueda de documentos utilizando la biblioteca LangChain y el servicio de vector store Pinecone.

El objetivo principal es poder realizar búsquedas sobre un conjunto de documentos de manera eficiente, aprovechando la capacidad de los modelos de lenguaje para codificar el texto en vectores semánticos.
## Comenzando

Estas instrucciones te ayudarán a obtener una copia del proyecto en funcionamiento en tu máquina local para fines de desarrollo y pruebas.

### Requisitos previos

Para ejecutar este proyecto, necesitarás tener los siguientes elementos instalados y configurados:

* **Python:** Se requiere Python 3.7 o una versión posterior. Puedes descargar Python desde el sitio web oficial: https://www.python.org/downloads/
* **Pip:** El administrador de paquetes de Python. Viene incluido con la instalación de Python.
* **Entorno virtual de Python:** Se recomienda crear y utilizar un entorno virtual de Python para aislar las dependencias del proyecto. Puedes usar venv o conda para crear y gestionar el entorno virtual.
* **OpenAI API Key:** Necesitarás una clave API de OpenAI para poder utilizar los servicios de embeddings de OpenAI. Puedes obtener una clave API registrándote en https://www.openai.com/.
* **Pinecone API Key:** Necesitarás una clave API de Pinecone para poder utilizar el servicio de vector store de Pinecone. Puedes registrarte y obtener las claves en https://www.pinecone.io/.

### Instalar y Ejecutar

1. Clona el repositorio:
    ```
    git clone https://github.com/AlexisGR117/AREP-TALLER9.git
    ```
2. Navega a la carpeta del proyecto:
    ```
    cd AREP-TALLER9
    ```
3. Crear y activar un entorno virtual de Python:
    ```
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
4. Instalar las dependencias del proyecto:
   ```
   pip install -r requirements.txt
   ```
5. Configurar las variables de entorno en cada uno de los scripts con tus llaves:
   ```
   os.environ["OPENAI_API_KEY"] = <Tu OpenAI API Key>
   os.environ["PINECONE_API_KEY"] = <Tu Pinecone API Key>
   ```
6. Ejecutar los scripts:
   ```
   python Challenge1.py
   ```
   ```
   python Challenge2.py
   ```
   ```
   python Challenge3.py
   ```

## Evaluación

### Challenge1:

```python
from langchain.chains import LLMChain
#from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

import os

os.environ["OPENAI_API_KEY"] = ""

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is at the core of Popper's theory of science?"

response = llm_chain.run(question)
print(response)
```

En este experimento se uso de la biblioteca LangChain para generar una respuesta a una pregunta dada utilizando el modelo de lenguaje de OpenAI. Aquí hay un desglose del código y el resultado:

1. El código configura el entorno importando los módulos necesarios de LangChain y LangChain Community, y configura la clave API de OpenAI.
2. Se define una plantilla de solicitud, que incluye un marcador de posición para la variable de entrada "question". Esta plantilla se utilizará para dar formato a la pregunta de entrada antes de pasarla al modelo de lenguaje.
3. Se crea una instancia del modelo de lenguaje OpenAI y se le asigna a la variable llm.
4. Se crea un LLMChain, que combina la plantilla de solicitud y el modelo de lenguaje. Esta cadena se utilizará para generar la respuesta a la pregunta.
5. La pregunta "What is at the core of Popper's theory of science?" se asigna a la variable pregunta.
6. Se llama al método ```llm_chain.run(question)``` para generar una respuesta a la pregunta dada.

El resultado de ejecutar este código es una respuesta que describe los aspectos clave de la teoría de la ciencia de Popper.

![Challenge1.png](img/Challenge1.png)

### Challenge2:

```python
import bs4
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

os.environ["OPENAI_API_KEY"] = ""


loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(splits[0])
print(splits[1])

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is Task Decomposition?")

print(response)
```
En este experimento se uso la biblioteca LangChain para implementar un sistema de pregunta-respuesta basado en un modelo de lenguaje (LLM) y un sistema de recuperación de documentos (Retriever). Aquí está un desglose del código y el resultado:

1. Se configuran las variables de entorno necesarias, incluida la clave API de OpenAI.
2. Se crea un cargador de documentos web (WebBaseLoader) que extrae el contenido de una página web específica (https://lilianweng.github.io/posts/2023-06-23-agent/).
3. Los documentos cargados se dividen en fragmentos más pequeños utilizando un SplitTextChunker, con un tamaño de fragmento de 1000 caracteres y un solapamiento de 200 caracteres.
4. Los fragmentos de texto se almacenan en un vector de búsqueda (Chroma) utilizando los embeddings de OpenAI, lo que permite la recuperación eficiente de fragmentos relevantes.
5. Se carga un prompt de pregunta-respuesta (rag-prompt) desde el repositorio de LangChain Hub.
6. Se crea una cadena de LLM que combina el contexto recuperado, la pregunta y el prompt, y utiliza el modelo de lenguaje ChatOpenAI para generar la respuesta.
7. Finalmente, se invoca la cadena con la pregunta "What is Task Decomposition?" y se imprime la respuesta generada.

El resultado de este código es una explicación concisa y precisa de qué es la Descomposición de Tareas, extrayendo la información relevante de los fragmentos de texto recuperados de la página web y generando una respuesta coherente utilizando el modelo de lenguaje.

![Challenge2.png](img/Challenge2.png)

### Challenge3:

```python
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec
import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["PINECONE_API_KEY"] = ""
os.environ["PINECONE_ENV"] = "gcp-starter"

def loadText():
    loader = TextLoader("Conocimiento.txt")
    documents = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
    )


    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    import pinecone


    index_name = "langchain-demo"
    pc = Pinecone(api_key='eb0f1c59-78f7-4e47-9017-87941c145474')

    print(pc.list_indexes())

    # First, check if our index already exists. If it doesn't, we create it
    if len(pc.list_indexes())==0:
        # we create a new index
        #pc.create_index(name=index_name, metric="cosine", dimension=1536)
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(
                environment=os.getenv("PINECONE_ENV"),
                pod_type="p1.x1",
                pods=1
            )
        )

    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

def search():
    embeddings = OpenAIEmbeddings()

    index_name = "langchain-demo"
    # if you already have an index, you can load it like this
    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

    query = "What is a saxophone?"
    docs = docsearch.similarity_search(query)

    print(docs[0].page_content)

loadText()
search()
```
En est eexperimento se implementa un sistema de búsqueda de documentos utilizando la biblioteca LangChain y el servicio de vector store Pinecone. Aquí está un desglose del código y el resultado:

1. Se configuran las variables de entorno necesarias, incluyendo la clave API de OpenAI y Pinecone.
2. La función loadText() se encarga de la carga y procesamiento del texto:
   * Se carga un archivo de texto llamado "Conocimiento.txt" utilizando el TextLoader de LangChain. 
   * El texto se divide en fragmentos más pequeños utilizando un RecursiveCharacterTextSplitter, con un tamaño de fragmento de 1000 caracteres y un solapamiento de 200 caracteres. 
   * Los fragmentos de texto se codifican como vectores utilizando los embeddings de OpenAI.
   * Los vectores se almacenan en un índice de Pinecone llamado "langchain-demo". Si el índice no existe, se crea uno nuevo. 
3. La función search() realiza la búsqueda de documentos:
   * Se carga el índice de Pinecone existente.
   * Se realiza una búsqueda de similitud utilizando la consulta "What is a saxophone?".
   * Se imprime el contenido de la página más relevante.

Al ejecutar este código, se cargarán los fragmentos de texto del archivo "Conocimiento.txt" en un índice de Pinecone y se realizará una búsqueda sobre ese índice utilizando la consulta "What is a saxophone?". El resultado debería ser una descripción del saxofono, extrayendo la información más relevante de los fragmentos de texto almacenados.

![Challenge3-1.png](img/Challenge3-1.png)

![Challenge3-2.png](img/Challenge3-2.png)

![Challenge3-3.png](img/Challenge3-3.png)

## Authors

* Jefer Alexis González Romero
