# Yaba College of Technology Student Management System



## 📌 Description
The **Yaba College of Technology Student Management System** is a web-based application designed to streamline student management for administrators. The system integrates AI-powered features such as facial recognition for authentication and AI-assisted text extraction from documents.

## 🚀 Features
- 🔑 **Admin Login & Signup with Facial Recognition**
- 🎓 **Student Management** (Add, Update, Delete student details)
- 📄 **Document Upload & AI-assisted Text Extraction**
- 📊 **Analytics Dashboard for Admin Activities**
- 🔐 **Secure Admin Account Deletion**

## 🛠 Technologies Used
- **Backend:** Python (Flask)
- **Frontend:** Streamlit
- **Database:** SQLite
- **AI & OCR:** OpenAI API, PyMuPDF, Tesseract OCR, Face Recognition
- **Libraries:** numpy, opencv-python, Pillow, dotenv, requests


## ⚙️ Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- pip (Python package manager)
- Required libraries (listed in `requirements.txt`)

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Haygee2/YCT-Student-Management-System.git
   cd YCT-Student-Management-System
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the environment variables by creating a `.env` file and adding:
   ```env
   DB_PATH=your_database_path.db
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Run the Flask server:
   ```bash
   python app.py
   ```
5. Start the Streamlit frontend:
   ```bash
   streamlit run frontend.py
   ```

## 📌 Usage
- Start the Flask backend and Streamlit frontend.
- Admin logs in using facial recognition.
- Admin can manage student records, upload documents, and view analytics.
- AI extracts text from uploaded documents for easy data retrieval.

## 🗄️ Database Schema

### Tables:
1. **Students** (id, name, matric\_number, department, face\_encoding, etc.)
2. **Admins** (id, username, password\_hash, face\_encoding, etc.)
3. **Documents** (id, student\_id, file\_path, extracted\_text, etc.)

## 🔑 Environment Variables
- `DB_PATH`: Path to the SQLite database.
- `OPENAI_API_KEY`: API key for AI-powered text extraction.

## 📷 Screenshots (Optional)
*Add screenshots here to showcase the application interface.*

## 🤝 Contributing
Pull requests are welcome! Feel free to fork the repo and submit PRs.


## 🙌 Acknowledgments
- OpenRouter for AI-powered text extraction.
- AI for document processing.
- Face Recognition library for authentication.

---

*Developed with ❤️ by HAYGEE*

