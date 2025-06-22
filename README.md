# Personalized EdTech Mentor

📚 **Personalized EdTech Mentor** is an interactive web application designed to help learners get the most out of their study materials. This app leverages AI to process educational PDFs, extract topics, generate quizzes, track progress, and provide personalized study recommendations—all in one place.

---

## Features

- **PDF Upload & Processing**: Upload your own educational PDF documents, which are analyzed and broken down into main topics and subtopics using AI.
- **Automatic Topic Extraction**: The app uses Large Language Models (LLMs) to extract key topics and subtopics, helping you organize your learning.
- **Interactive Topic Selection**: Choose which topic to study from your uploaded material.
- **Conversational Mentor**: Ask questions on any topic and get AI-generated answers, creating a personalized learning experience.
- **Quiz Generation**: For each topic, the system generates multiple-choice quizzes to help reinforce your understanding.
- **Progress Tracking**: Monitor your status, average quiz scores, and attempts for each topic.
- **Personalized Recommendations**: Receive actionable study advice based on your quiz performance and progress.
- **Persistent Storage**: All data, including PDFs, extracted topics, quiz results, and recommendations, are saved for future sessions.

---

## How It Works

1. **Upload a PDF**: Use the sidebar to upload an educational PDF. The system processes the document and extracts topics.
2. **Choose a Topic**: Select a topic from the extracted list to start studying.
3. **Ask Questions**: Use the chat interface to ask questions and get instant explanations.
4. **Take Quizzes**: Challenge yourself with AI-generated quizzes tailored to the topic.
5. **Get Recommendations**: Based on your quiz performance, receive targeted study tips to improve your understanding.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chittoorking/personalized-edtech-mentor.git
   cd personalized-edtech-mentor
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create `.env` and add your OpenAI (or other LLM) API keys as required.

5. **Run the app**
   ```bash
   streamlit run personalized_edtech_mentor.py
   ```

---

## Requirements

- Python 3.8+
- See `requirements.txt` for Python package dependencies.
- OpenAI API key (or other supported LLM/provider)
- [Streamlit](https://streamlit.io/) for the web interface

---

## Technologies Used

- **Streamlit**: Rapid web app framework for Python
- **LangChain**: Framework for developing applications powered by language models
- **OpenAI / HuggingFace**: LLMs for language understanding and response generation
- **ChromaDB**: Persistent vector store for semantic search
- **PDF Processing**: `unstructured`, `pdfminer.six`, `pdf2image`, etc.

---

## Project Structure

```
personalized-edtech-mentor/
├── personalized_edtech_mentor.py   # Main application
├── requirements.txt                # Python dependencies
├── data/                           # Uploaded PDFs and vector stores
├── .env                    # Environment variables
└── README.md
```

---

## Usage Tips

- Upload only educational PDFs for best results.
- Quiz and recommendations are generated per topic; take quizzes often for personalized feedback.
- Your progress is saved locally in the `data/` directory.

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI](https://openai.com/)
- [HuggingFace](https://huggingface.co/)
