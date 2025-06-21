# Importing necessary libraries
import streamlit as st
import os
import json
import tempfile
import shutil
from typing import Dict, List, Optional
from pathlib import Path
import openai

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="Personalized Education Mentor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

class EducationBot:
    def __init__(self):
        self.vector_store = None
        self.topics = {}
        self.user_progress = {}
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4")
        # Initialize embeddings with model and normalize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        # Do NOT clear existing Chroma DB on every init
        # vector_store_path = DATA_DIR / "chroma_db"
        # if vector_store_path.exists():
        #     import shutil
        #     shutil.rmtree(vector_store_path)
        self.load_existing_pdfs()

    def load_existing_pdfs(self):
        """Load any existing PDFs from the data directory"""
        if not DATA_DIR.exists():
            return
        # Load vector store if it exists
        vector_store_path = DATA_DIR / "chroma_db"
        if vector_store_path.exists():
            try:
                self.vector_store = Chroma(
                    persist_directory=str(vector_store_path),
                    embedding_function=self.embeddings
                )
            except Exception as e:
                st.error(f"Error loading vector store: {str(e)}")
                self.vector_store = None
        # Load topics if they exist
        topics_path = DATA_DIR / "topics.json"
        if topics_path.exists():
            try:
                with open(topics_path, 'r') as f:
                    self.topics = json.load(f)
            except Exception as e:
                st.error(f"Error loading topics: {str(e)}")
                self.topics = {}

    def save_state(self):
        """Save the current state (vector store and topics)"""
        if self.vector_store:
            self.vector_store.persist()
        if self.topics:
            topics_path = DATA_DIR / "topics.json"
            with open(topics_path, 'w') as f:
                json.dump(self.topics, f)

    def get_available_pdfs(self) -> List[str]:
        """Get list of available PDFs"""
        if not DATA_DIR.exists():
            return []
        return [f.name for f in DATA_DIR.glob("*.pdf")]
        
    def _validate_topics_json(self, topics_json: str) -> Optional[dict]:
        """Helper function to validate and clean up topics JSON"""
        try:
            # Clean up whitespace and find valid JSON substring if needed
            topics_json = topics_json.strip()
            if not topics_json.startswith("{"):
                start = topics_json.find("{")
                end = topics_json.rfind("}") + 1
                if start >= 0 and end > start:
                    topics_json = topics_json[start:end]
            
            topics_dict = json.loads(topics_json)
            
            # Validate dictionary structure
            if not isinstance(topics_dict, dict):
                st.error("Invalid topics format: Expected a dictionary")
                return None
                
            # Validate each topic has a list of subtopics
            validated_topics = {}
            for topic, subtopics in topics_dict.items():
                if isinstance(subtopics, list):
                    # Remove any empty subtopics and strip whitespace
                    clean_subtopics = [s.strip() for s in subtopics if s.strip()]
                    if clean_subtopics:  # Only include topics with valid subtopics
                        validated_topics[topic.strip()] = clean_subtopics
                
            return validated_topics if validated_topics else None
            
        except json.JSONDecodeError as je:
            st.error(f"Error parsing topics JSON: {str(je)}")
            st.text("Raw response:")
            st.code(topics_json)
            return None
            
    def process_pdf(self, pdf_file, filename: str = None) -> Dict:
        """Process uploaded PDF and extract topics"""
        pdf_path = None
        try:
            # Save PDF to data directory
            if filename is None:
                filename = pdf_file.name
            pdf_path = DATA_DIR / filename
            
            # Save uploaded file
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())
                
            # Load and process PDF
            loader = DirectoryLoader(str(DATA_DIR), glob="*.pdf")
            pages = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(pages)
            
            # Create or update vector store with HuggingFace embeddings
            vector_store_path = DATA_DIR / "chroma_db"
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=str(vector_store_path)
                )
            else:
                self.vector_store.add_documents(splits)
                self.vector_store.persist()
                
            # Extract topics using LLM
            topic_prompt = PromptTemplate(
                template="""Extract the main topics and subtopics from this text. 
                Please format the response EXACTLY as a JSON object where keys are main topics 
                and values are arrays of subtopics. Focus on educational content only.
                
                Example format:
                {{
                    "Main Topic 1": ["Subtopic 1.1", "Subtopic 1.2"],
                    "Main Topic 2": ["Subtopic 2.1", "Subtopic 2.2"]
                }}
                
                Text: {text}
                
                Response (must be valid JSON):""",
                input_variables=["text"]
            )
            
            # Create extraction chain using the new pattern
            topic_chain = topic_prompt | self.llm | StrOutputParser()
            
            # Process each chunk to identify topics
            topics_found = False
            for split in splits[:5]:  # Process first 5 chunks for topics
                try:
                    topics_json = topic_chain.invoke({"text": split.page_content})
                    validated_topics = self._validate_topics_json(topics_json)
                    
                    if validated_topics:
                        self.topics.update(validated_topics)
                        topics_found = True
                        
                except Exception as e:
                    st.warning(f"Error processing chunk for topics: {str(e)}")
                    continue
            
            if not topics_found:
                st.warning("No valid topics were extracted. The PDF might not contain well-structured educational content.")
            
            # Save the current state
            self.save_state()
            return {"status": "success", "message": "PDF processed successfully"}
            
        except Exception as e:
            if pdf_path and pdf_path.exists():
                pdf_path.unlink()  # Remove the PDF if processing failed
            return {"status": "error", "message": str(e)}

    def generate_quiz(self, topic: str) -> List[Dict]:
        """Generate quiz questions for a specific topic, filtering out meta-information."""
        if not self.vector_store:
            return []
        
        # Get relevant content from vector store 
        # Retrieve more chunks for better filtering
        results = self.vector_store.similarity_search(topic, k=5)  
        # Filter out meta-information (author, table of contents, etc.)
        meta_keywords = ["author", "professor", "table of contents", "contents", "copyright", "isbn", "publisher"]
        filtered_results = []
        for doc in results:
            content_lower = doc.page_content.lower()
            if not any(keyword in content_lower for keyword in meta_keywords):
                filtered_results.append(doc)
        # If all chunks are filtered, fall back to original results
        if not filtered_results:
            filtered_results = results
        content = " ".join([doc.page_content for doc in filtered_results])
        quiz_prompt = PromptTemplate(
            template="""Based on this content about {topic}:
{content}

Generate 3 multiple choice questions that test understanding of the educational concepts, NOT about authors, publication info, or table of contents. Format as JSON array with objects containing:
- question (string)
- choices (array of 4 strings)
- correct_answer (index of correct option)
- explanation (string explaining the correct answer)
- difficulty (number 1-3)
""",
            input_variables=["topic", "content"]
        )
        # Create quiz chain using the new pattern
        quiz_chain = quiz_prompt | self.llm | StrOutputParser()
        quiz_json = quiz_chain.invoke({"topic": topic, "content": content})
        return json.loads(quiz_json)

    def get_topic_content(self, topic: str) -> str:
        """Retrieve relevant content for a topic from vector store"""
        if self.vector_store:
            results = self.vector_store.similarity_search(topic, k=2)
            return " ".join([doc.page_content for doc in results])
        return ""

    def evaluate_response(self, topic: str, score: float) -> None:
        """Update user progress for a topic"""
        if topic not in self.user_progress:
            self.user_progress[topic] = {
                'scores': [],
                'status': 'not_started',
                'attempts': 0
            }
        
        progress = self.user_progress[topic]
        progress['scores'].append(score)
        progress['attempts'] += 1
        avg_score = sum(progress['scores']) / len(progress['scores'])
        
        if avg_score >= 0.8:
            progress['status'] = 'mastered'
        elif avg_score >= 0.6:
            progress['status'] = 'clear'
        else:
            progress['status'] = 'needs_review'

    def get_study_recommendations(self, topic: str) -> List[str]:
        """Generate personalized study recommendations based on progress"""
        if topic not in self.user_progress:
            return ["Complete a quiz to get personalized recommendations"]
        
        progress = self.user_progress[topic]
        content = self.get_topic_content(topic)
        
        prompt = PromptTemplate(
            template="""Based on this student's performance:
            Topic: {topic}
            Average Score: {avg_score}
            Status: {status}
            Attempts: {attempts}
            
            And this content about the topic:
            {content}
            
            Provide 3 specific study recommendations to help them improve.
            Focus on areas they might be struggling with and suggest specific resources or approaches.""",
            input_variables=["topic", "avg_score", "status", "attempts", "content"]
        )
        
        # Create recommendation chain using the new pattern
        chain = prompt | self.llm | StrOutputParser()
        avg_score = sum(progress['scores']) / len(progress['scores'])
        
        recommendations = chain.invoke({
            "topic": topic,
            "avg_score": avg_score,
            "status": progress['status'],
            "attempts": progress['attempts'],
            "content": content
        })
        return recommendations.split("\n")

# Initialize session state
if "bot" not in st.session_state:
    st.session_state.bot = EducationBot()
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quiz_active" not in st.session_state:
    st.session_state.quiz_active = False
    st.session_state.quiz_answers = []

def get_openai_response(user_input):
    if not st.session_state.bot.vector_store:
        return "Please process a PDF first to enable the question-answering feature."
    
    retriever = st.session_state.bot.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Get top 3 most relevant chunks
    )
    
    # Build conversational history for context-aware retrieval and LLM
    # Use last 6 exchanges for more context
    chat_history = st.session_state.chat_history[-6:]  
    conversation = "\n".join([
        f"Student: {q}\nMentor: {a}" for q, a in chat_history
    ])
    
    # For retrieval, combine user input with recent chat for better context
    retrieval_query = user_input
    if chat_history:
        retrieval_query = chat_history[-1][0] + " " + user_input
    
    # Get relevant content for the current question and context
    retrieved_docs = retriever.get_relevant_documents(retrieval_query)
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = PromptTemplate(
        template="""You are an expert educational mentor. Use the following learning materials and the ongoing conversation to answer the student's next message.\n\nIf the student's message refers to something discussed earlier, respond in a conversational way, referencing previous answers as needed. Do not repeat the question. If you can't find a definitive answer in the provided content, say so politely and suggest what related topics they might want to explore instead.\n\nKeep your responses:\n1. Educational and informative\n2. Clear and well-structured\n3. Conversational and context-aware\n4. Focused on accuracy rather than speculation\n5. Encouraging further exploration when appropriate\n\nContext from learning materials:\n{context}\n\nConversation so far:\n{chat_history}\n\nStudent's next message: {question}\n\nYour response:""",
        input_variables=["context", "chat_history", "question"]
    )
    
    chat = ChatOpenAI(temperature=0.5, model="gpt-4")
    chain = prompt | chat | StrOutputParser()
    
    response = chain.invoke({
        "context": context,
        "chat_history": conversation,
        "question": user_input
    })
    
    return response

def main():
    st.title("📚 Personalized Education Mentor")
    
    # Sidebar for file upload and topic selection
    with st.sidebar:
        st.header("Learning Materials")
        
        # Show available PDFs
        st.subheader("Available PDFs")
        available_pdfs = st.session_state.bot.get_available_pdfs()
        if available_pdfs:
            for pdf in available_pdfs:
                st.text(f"📄 {pdf}")
        else:
            st.info("No PDFs uploaded yet")
        
        st.divider()
        
        # Upload new PDF
        st.subheader("Upload New Material")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    # Check if PDF already exists
                    if uploaded_file.name in available_pdfs:
                        st.warning("This PDF has already been processed!")
                    else:
                        result = st.session_state.bot.process_pdf(uploaded_file)
                        if result["status"] == "success":
                            st.success("PDF processed successfully!")
                            st.rerun()
                        else:
                            st.error(f"Error: {result['message']}")
        
        if st.session_state.bot.topics:
            st.header("Topics")
            topics = list(st.session_state.bot.topics.keys())
            selected_topic = st.selectbox("Select a topic to study", topics)
            if selected_topic != st.session_state.current_topic:
                st.session_state.current_topic = selected_topic
                st.session_state.quiz_active = False
                st.session_state.chat_history = []  # Reset chat history when topic changes
                st.session_state["question_input"] = ""  # Reset question input when topic changes
    
    # Main content area
    if not st.session_state.bot.topics:
        st.info("👆 Please upload a PDF document to start learning")
        return
    
    # Topic content and interaction area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📖 {st.session_state.current_topic}")
        
        # Chat interface
        st.header("💭 Ask Questions")
        def handle_question():
            user_question = st.session_state["question_input"]
            if user_question:
                with st.spinner("Thinking..."):
                    response = get_openai_response(user_question)
                    st.session_state.chat_history.append((user_question, response))
                st.session_state["question_input"] = ""  # Clear input after processing
        user_question = st.text_input(
            "Ask a question about this topic",
            key="question_input",
            on_change=handle_question
        )
        # Display chat history
        for q, a in st.session_state.chat_history:
            st.write(f"**Q:** {q}")
            st.write(f"**A:** {a}")
            st.divider()
    
    with col2:
        st.header("📊 Progress")
        if st.session_state.current_topic in st.session_state.bot.user_progress:
            progress = st.session_state.bot.user_progress[st.session_state.current_topic]
            st.metric("Status", progress["status"].replace("_", " ").title())
            if progress["scores"]:
                avg_score = sum(progress["scores"]) / len(progress["scores"])
                st.metric("Average Score", f"{avg_score:.2%}")
                st.metric("Attempts", progress["attempts"])
        
        # Quiz section
        st.header("📝 Quiz")
        if not st.session_state.quiz_active:
            if st.button("Start Quiz"):
                st.session_state.quiz_active = True
                st.session_state.current_quiz = st.session_state.bot.generate_quiz(st.session_state.current_topic)
                st.session_state.quiz_answers = [None] * len(st.session_state.current_quiz)  # Initialize with None
                st.rerun()
        else:
            if "current_quiz" in st.session_state:
                for i, q in enumerate(st.session_state.current_quiz):
                    st.subheader(f"Question {i + 1}")
                    st.write(q["question"])
                    st.session_state.quiz_answers[i] = st.radio(
                        "Choose your answer:",
                        q["choices"],
                        key=f"q_{i}"
                    )
                if st.button("Submit Quiz"):
                    # Calculate score only on submit
                    correct = [
                        (ans is not None and q["choices"].index(ans) == q["correct_answer"])
                        for ans, q in zip(st.session_state.quiz_answers, st.session_state.current_quiz)
                    ]
                    score = sum(correct) / len(correct)
                    st.session_state.bot.evaluate_response(st.session_state.current_topic, score)
                    # Show results
                    st.write(f"Your score: {score:.2%}")
                    for i, (q, is_correct) in enumerate(zip(st.session_state.current_quiz, correct)):
                        if is_correct:
                            st.success(f"Question {i + 1}: Correct!")
                        else:
                            st.error(f"Question {i + 1}: Incorrect")
                            st.info(f"Explanation: {q['explanation']}")
                    st.session_state.quiz_active = False
                    # Show recommendations
                    st.header("📈 Recommendations")
                    recommendations = st.session_state.bot.get_study_recommendations(st.session_state.current_topic)
                    for rec in recommendations:
                        st.write(f"• {rec}")

if __name__ == "__main__":
    main()
