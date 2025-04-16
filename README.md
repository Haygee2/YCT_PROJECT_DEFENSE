Student Management System

A dual-version application for managing student records, with both basic and AI-enhanced functionality.

 📌 Overview

- flask_app.py: Basic student management system with authentication and file handling
- app.py: Enhanced version with AI-powered document processing features

 🌟 Features Comparison

| Feature                |  flask_app.py   |    app.py       |
|------------------------|-----------------|----------------|
| User Authentication    | ✅             | ✅              |
| Facial Recognition     | ✅             | ✅              |
| Student Data Upload    | ✅             | ✅              |
| File Storage           | ✅             | ✅              |
| AI Document Processing | ❌             | ✅              |
| Text Extraction        | ❌             | ✅              |
| Smart Search           | ❌             | ✅              |

 🚀 Getting Started

📒 Prerequisites
- Python 3.8+
- Streamlit
- SQLite3
- For app.py only:
  - PyPDF2 (pip install pypdf2)
  - python-docx (pip install python-docx)
  - Other AI dependencies e.g OPENROUTER, OPENAI

Installation
```bash
git clone [your-repo-url]
cd student-management-system
pip install -r requirements.txt
```

## 🖥️ Running the Applications

### Basic Version (`flask_app.py`)
```bash
streamlit run flask_app.py
```

### AI-Enhanced Version (`app.py`)
```bash
streamlit run app.py
```

## 🔧 Configuration

Create a `.env` file for both versions:
```env
# Database configuration
DB_NAME=students.db
ADMIN_DB=admins.db

# For app.py only
AI_MODEL_PATH=models/document_processor
```

## 📂 Project Structure

```
student-management/
├── students_data/          # Student files storage
├── admins.db               # Admin credentials database
├── flask_app.py            # Basic version
├── app.py                  # AI-enhanced version
├── requirements.txt        # Dependencies
└── README.md
```

## 🤖 AI Features (app.py only)

The enhanced version includes:
- Automated text extraction from PDFs/DOCs
- Document content analysis
- Smart search across document contents
- Metadata generation for uploaded files

## 📜 License
MIT License

## 📧 Contact
[adebowalegoodness688@gmail.com] | [github.com/Haygee2/YCT_PROJECT_DEFENSE]
```
