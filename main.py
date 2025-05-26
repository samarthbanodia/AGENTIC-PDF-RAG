import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from crewai import Crew, Task, Agent

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key='sk-proj-...',
            model="text-embedding-3-small"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )

    def extract_text(self, documents):
        return ''.join(page.extract_text() for doc in documents 
                      for page in PdfReader(doc).pages)

    def process_content(self, text):
        chunks = self.text_splitter.create_documents([text])
        return FAISS.from_documents(chunks, self.embeddings)

class AIAgents:
    def __init__(self):
        self.info_agent = Agent(
            role='Data Extractor',
            goal='Identify key entities and relationships',
            llm=self._create_llm(),
            verbose=True
        )
        
        self.synth_agent = Agent(
            role='Data Organizer',
            goal='Structure extracted information',
            llm=self._create_llm(),
            verbose=True
        )
        
        self.query_agent = Agent(
            role='Query Handler', 
            goal='Answer document questions',
            llm=self._create_llm(),
            verbose=True
        )

    def _create_llm(self):
        return {
            'provider': "openai",
            'config': {
                'model': "gpt-3.5-turbo",
                'api_key': 'sk-proj-...',
                'timeout': 60
            }
        }

    def create_crew(self):
        return Crew(
            agents=[self.info_agent, self.synth_agent, self.query_agent],
            tasks=self._create_tasks(),
            verbose=True
        )

    def _create_tasks(self):
        return [
            Task(
                description="Extract entities from: {text}",
                agent=self.info_agent,
                expected_output="JSON with entities/relationships"
            ),
            Task(
                description="Organize extracted data",
                agent=self.synth_agent,
                expected_output="Structured JSON output"
            ),
            Task(
                description="Answer query: {question}",
                agent=self.query_agent,
                expected_output="Concise English response"
            )
        ]

class ChatInterface:
    def __init__(self):
        self.processor = PDFProcessor()
        self.agents = AIAgents()
        
        if 'history' not in st.session_state:
            st.session_state.history = []
            
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None

    def run(self):
        st.title("Document Analysis System")
        self._handle_file_upload()
        self._handle_queries()

    def _handle_file_upload(self):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            text = self.processor.extract_text([uploaded_file])
            st.session_state.vector_store = self.processor.process_content(text)
            st.success("Document processed successfully")

    def _handle_queries(self):
        query = st.text_input("Ask about the document:")
        if query and st.session_state.vector_store:
            self._process_query(query)

    def _process_query(self, question):
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
        context = "\n".join(doc.page_content for doc in retriever.invoke(question))
        
        crew = self.agents.create_crew()
        result = crew.kickoff(inputs={
            "text": context,
            "question": question
        })
        
        st.session_state.history.append(f"Q: {question}\nA: {result}")
        st.write(f"Response: {result}")

if __name__ == "__main__":
    interface = ChatInterface()
    interface.run()
