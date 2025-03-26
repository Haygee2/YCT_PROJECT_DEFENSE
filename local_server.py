from flask import Flask, request, jsonify
import os
import sqlite3
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import cv2
import numpy as np
from datetime import datetime
import streamlit as st
import face_recognition
from dotenv import load_dotenv
import openai  # Ensure you install the OpenAI library: pip install openai
import requests  # Ensure you install the requests library: pip install requests
from chatbot import chat_with_ai  # Import the chatbot function

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

DB_PATH = os.getenv("DB_PATH", "students.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openai.api_key = OPENAI_API_KEY

def execute_query(query, params=(), fetchone=False, fetchall=False):
    """General function to execute queries safely."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            if fetchone:
                return cursor.fetchone()
            elif fetchall:
                return cursor.fetchall()
            
            conn.commit()
            return None
    except sqlite3.Error as e:
        return {"error": str(e)}

def init_db():
    """Initialize the database and create necessary tables."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                                username TEXT PRIMARY KEY,
                                password TEXT,
                                role TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                                matric_number TEXT PRIMARY KEY,
                                name TEXT,
                                folder TEXT,
                                email TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS activity_logs (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                user TEXT,
                                action TEXT,
                                timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS document_versions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                matric_number TEXT,
                                document_name TEXT,
                                version INTEGER,
                                file_path TEXT,
                                text_file_path TEXT,
                                timestamp TEXT)''')
            conn.commit()
            print("Database initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")

# Ensure the database is initialized at the start
init_db()

@app.route('/students', methods=['GET'])
def get_all_students():
    """Fetch all students from the database."""
    query = "SELECT matric_number, name FROM students"
    students = execute_query(query, fetchall=True)
    return jsonify(students)

@app.route('/student/<matric_number>', methods=['GET'])
def get_student_info(matric_number):
    """Fetch student details by matric_number."""
    query = "SELECT matric_number, name, folder, face_image, face_encoding_path, email FROM students WHERE matric_number = ?"
    student = execute_query(query, (matric_number,), fetchone=True)
    if student:
        return jsonify(student)
    return jsonify({"error": "Student not found"}), 404

@app.route('/upload', methods=['POST'])
def upload_document():
    """Handle document upload and text extraction."""
    matric_number = request.form.get('matric_number')
    name = request.form.get('name')
    file = request.files.get('file')

    if not matric_number or not name or not file:
        return jsonify({"error": "Missing required fields"}), 400

    folder = os.path.join("students_data", f"{name}_{matric_number[-3:]}")
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)
    file.save(file_path)

    text = ""
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = Image.open(file_path)
        text = extract_text_from_image(image)

    text_file_path = file_path.rsplit('.', 1)[0] + ".txt"
    with open(text_file_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text)

    save_document_version(matric_number, file.filename, file_path, text_file_path)
    return jsonify({"message": "Document uploaded successfully", "file_path": file_path, "text_file_path": text_file_path})

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    preprocessed_image = Image.fromarray(sharpened)
    return pytesseract.image_to_string(preprocessed_image)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF."""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += extract_text_from_image(img)
    return text

def save_document_version(matric_number, document_name, file_path, text_file_path):
    """Save a new version of the document."""
    version = get_latest_document_version(matric_number, document_name) + 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = '''INSERT INTO document_versions (matric_number, document_name, version, file_path, text_file_path, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)'''
    execute_query(query, (matric_number, document_name, version, file_path, text_file_path, timestamp))

def get_latest_document_version(matric_number, document_name):
    """Get the latest version number of a document for a student."""
    query = '''SELECT MAX(version) FROM document_versions WHERE matric_number = ? AND document_name = ?'''
    result = execute_query(query, (matric_number, document_name), fetchone=True)
    return result[0] if result and result[0] is not None else 0

@app.route('/login', methods=['POST'])
def login():
    """Handle user login."""
    data = request.json
    matric_number = data.get('username')  # Matric number as username
    surname = data.get('password')  # Surname as password

    # Fetch student details
    query = "SELECT matric_number, name, folder FROM students WHERE matric_number = ?"
    student = execute_query(query, (matric_number,), fetchone=True)

    if student:
        stored_surname = student[1].split()[-1]  # Extract surname from name
        if stored_surname.lower() == surname.lower():
            # Perform facial recognition verification
            folder = student[2]
            stored_encoding_path = os.path.join(folder, "face_encoding.npy")
            if os.path.exists(stored_encoding_path):
                if capture_and_verify_face(folder, stored_encoding_path):
                    return jsonify({"message": f"Welcome, {student[1]}!", "role": "Student"})
                else:
                    return jsonify({"error": "Facial recognition failed"}), 401
            else:
                return jsonify({"error": "No stored face encoding found"}), 404
        else:
            return jsonify({"error": "Invalid credentials"}), 401
    return jsonify({"error": "Student not found"}), 404

@app.route('/signup', methods=['POST'])
def signup():
    """Handle user signup."""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')
    query = "INSERT INTO users (username, password, role) VALUES (?, ?, ?)"
    execute_query(query, (username, password, role))
    return jsonify({"message": "Account created successfully"})

@app.route('/student', methods=['POST'])
def add_student():
    """Add or update student information."""
    data = request.json
    matric_number = data.get('matric_number')
    name = data.get('name')
    email = data.get('email', "")
    folder = os.path.join("students_data", f"{name}_{matric_number[-3:]}")
    os.makedirs(folder, exist_ok=True)
    query = '''INSERT OR REPLACE INTO students (matric_number, name, folder, email) 
               VALUES (?, ?, ?, ?)'''
    execute_query(query, (matric_number, name, folder, email))
    return jsonify({"message": "Student added/updated successfully"})

@app.route('/student/<matric_number>/documents', methods=['GET'])
def list_documents(matric_number):
    """List all documents for a student."""
    student = execute_query("SELECT folder FROM students WHERE matric_number = ?", (matric_number,), fetchone=True)
    if not student:
        return jsonify({"error": "Student not found"}), 404
    folder = student[0]
    if not os.path.exists(folder):
        return jsonify([])
    documents = [f for f in os.listdir(folder) if f.endswith(('.txt', '.pdf', '.jpg', '.jpeg', '.png'))]
    return jsonify(documents)

@app.route('/student/<matric_number>/document', methods=['POST'])
def upload_student_document(matric_number):
    """Upload a document for a student."""
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    student = execute_query("SELECT folder FROM students WHERE matric_number = ?", (matric_number,), fetchone=True)
    if not student:
        return jsonify({"error": "Student not found"}), 404
    folder = student[0]
    file_path = os.path.join(folder, file.filename)
    file.save(file_path)
    text = ""
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = Image.open(file_path)
        text = extract_text_from_image(image)
    text_file_path = file_path.rsplit('.', 1)[0] + ".txt"
    with open(text_file_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text)
    save_document_version(matric_number, file.filename, file_path, text_file_path)
    return jsonify({"message": "Document uploaded successfully", "file_path": file_path, "text_file_path": text_file_path})

@app.route('/activity_logs', methods=['POST'])
def log_activity():
    """Log user activity."""
    data = request.json
    user = data.get('user')
    action = data.get('action')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = "INSERT INTO activity_logs (user, action, timestamp) VALUES (?, ?, ?)"
    execute_query(query, (user, action, timestamp))
    return jsonify({"message": "Activity logged successfully"})

@app.route('/ai/extract_text', methods=['POST'])
def ai_extract_text():
    """Extract text from an image using OpenRouter AI."""
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        # Use OpenRouter API to process the question
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4",  # Specify the model you want to use
            "messages": [{"role": "user", "content": question}]
        }
        response = requests.post("https://api.openrouter.ai/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

        answer = response.json()["choices"][0]["message"]["content"].strip()
        return jsonify({"question": question, "answer": answer})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {e}"}), 500
    except KeyError:
        return jsonify({"error": "Unexpected response format from OpenRouter API"}), 500

def capture_and_verify_face(student_folder, stored_image_path):
    """Capture a face and verify it against the stored image."""
    st.title("Facial Recognition Verification")

    # Capture face using Streamlit's camera input
    img_file = st.camera_input("Capture your face for verification")

    if img_file:
        try:
            # Save the captured image temporarily
            temp_image_path = os.path.join(student_folder, "temp_captured_face.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(img_file.getvalue())

            # Load the captured image
            captured_image = face_recognition.load_image_file(temp_image_path)

            # Detect face encodings in the captured image
            captured_encodings = face_recognition.face_encodings(captured_image)

            if len(captured_encodings) > 0:
                captured_encoding = captured_encodings[0]

                # Load the stored face image and its encoding
                if os.path.exists(stored_image_path):
                    stored_image = face_recognition.load_image_file(stored_image_path)
                    stored_encodings = face_recognition.face_encodings(stored_image)

                    if len(stored_encodings) > 0:
                        stored_encoding = stored_encodings[0]

                        # Compare the captured encoding with the stored encoding
                        matches = face_recognition.compare_faces([stored_encoding], captured_encoding)
                        if matches[0]:
                            st.success("Face verified successfully!")
                            return True
                        else:
                            st.error("Face verification failed. Please try again.")
                    else:
                        st.error("No face detected in the stored image.")
                else:
                    st.error("No stored face image found for this student.")
            else:
                st.error("No face detected in the captured image. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Clean up the temporary image file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    else:
        st.info("Please capture your face for verification.")
    return False

def streamlit_frontend():
    """Streamlit frontend for interacting with the backend."""
    st.title("YABA COLLEGE OF TECHNOLOGY COMPUTER ENGINEERING DEPARTMENT")

    st.sidebar.title("Navigation")
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "verified_student" not in st.session_state:
        st.session_state.verified_student = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None

    if not st.session_state.logged_in:
        page = st.sidebar.radio("Go to", ["Login", "Sign Up"], key="main_nav")
    else:
        if st.session_state.user_role == "Admin":
            page = st.sidebar.radio("Go to", ["Admin Panel", "Manage Students", "Analytics Dashboard"], key="admin_nav")
        else:
            page = st.sidebar.radio("Go to", ["Student Panel", "AI Study Helper"], key="student_nav")

    if page == "Login":
        st.subheader("Login")
        role = st.selectbox("Select Role:", ["Admin", "Student"], key="role_selectbox")
        username = st.text_input("Username:", key="username_input")
        password = st.text_input("Password:", type="password", key="password_input")
        
        if st.button("Login", key="login_button"):
            if role == "Student":
                # Fetch student details
                student = execute_query("SELECT matric_number, name, folder FROM students WHERE matric_number = ?", (username,), fetchone=True)
                if student:
                    # Extract surname as the first name
                    full_name = student[1].strip()
                    surname = full_name.split()[0].lower()  # Extract the first name and convert to lowercase
                    if surname == password.lower():  # Compare in a case-insensitive manner
                        st.session_state.logged_in = True
                        st.session_state.user_role = "Student"
                        st.session_state.verified_student = (student[0], student[1])  # Set verified student
                        st.success("Student Login Successful!")
                        execute_query("INSERT INTO activity_logs (user, action, timestamp) VALUES (?, ?, ?)", 
                                      (username, "Student Login", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    else:
                        st.error("Invalid credentials. Please check your first name.")
                else:
                    st.error("Student not found. Please check your matric number.")
            else:
                # Admin login logic remains unchanged
                user = execute_query("SELECT username, password, role FROM users WHERE username = ?", (username,), fetchone=True)
                if user and user[1] == password and user[2] == role:
                    st.session_state.logged_in = True
                    st.session_state.user_role = "Admin"
                    st.success("Admin Login Successful!")
                    execute_query("INSERT INTO activity_logs (user, action, timestamp) VALUES (?, ?, ?)", 
                                  (username, "Admin Login", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                else:
                    st.error("Invalid credentials or role mismatch.")

    if page == "Sign Up":
        st.subheader("Sign Up")
        new_username = st.text_input("New Username:", key="new_username_input")
        new_password = st.text_input("New Password:", type="password", key="new_password_input")
        new_role = st.radio("Select Role:", ["Admin", "Student"], key="new_role_radio")
        
        if st.button("Create Account", key="create_account_button"):
            existing_user = execute_query("SELECT username FROM users WHERE username = ?", (new_username,), fetchone=True)
            if existing_user:
                st.error("Username already exists. Please choose a different username.")
            else:
                execute_query("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                              (new_username, new_password, new_role))
                st.success("Account created successfully! You can now log in.")

    if st.session_state.logged_in:
        if st.sidebar.button("Log Out", key="logout_sidebar"):
            st.session_state.logged_in = False
            st.session_state.verified_student = None
            st.session_state.user_role = None
            st.experimental_set_query_params(logged_in=False)

    if page == "Admin Panel" and st.session_state.user_role == "Admin":
        st.subheader("Admin Panel: Manage Students and Process Documents with AI")
        
        # Section to save student details
        st.subheader("Save Student Details")
        admin_matric = st.text_input("Enter Student Matric Number:", key="admin_matric").strip()
        admin_name = st.text_input("Enter Student Name:", key="admin_name")
        admin_email = st.text_input("Enter Student Email:", key="admin_email")
        admin_phone = st.text_input("Enter Student Phone Number:", key="admin_phone")

        if st.button("Save Student Details", key="save_student_details_button"):
            if admin_matric and admin_name and admin_email and admin_phone:
                folder = os.path.join("students_data", f"{admin_name}_{admin_matric[-3:]}")
                os.makedirs(folder, exist_ok=True)
                query = '''INSERT OR REPLACE INTO students (matric_number, name, folder, email) 
                           VALUES (?, ?, ?, ?)'''
                execute_query(query, (admin_matric, admin_name, folder, admin_email))
                st.success(f"Student details saved for {admin_name} (Matric: {admin_matric}).")
            else:
                st.error("Please fill in all the fields.")

        # Section for AI-assisted document upload and processing
        st.subheader("Upload and Process Documents with AI")
        admin_doc_file = st.file_uploader("Upload a document (PDF, JPG, PNG) for AI-assisted text extraction:", type=["pdf", "jpg", "jpeg", "png"], key="admin_doc_file")
        if admin_doc_file:
            file_path = os.path.join("admin_uploaded_files", admin_doc_file.name)
            os.makedirs("admin_uploaded_files", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(admin_doc_file.read())
            st.success(f"File '{admin_doc_file.name}' uploaded successfully!")

            # Extract text using AI
            if st.button("Extract Text with AI", key="extract_text_ai_button"):
                extracted_text = chat_with_ai("Extract detailed text from this document:", file_path=file_path)
                st.subheader("AI-Extracted Text:")
                st.write(extracted_text)  # Display the extracted text as non-editable content

                # Option to save the extracted text
                if st.button("Save Extracted Text", key="save_extracted_text_button"):
                    if admin_matric and admin_name:
                        folder = os.path.join("students_data", f"{admin_name}_{admin_matric[-3:]}")
                        os.makedirs(folder, exist_ok=True)
                        text_file_path = os.path.join(folder, f"{admin_doc_file.name.rsplit('.', 1)[0]}_ai_extracted.txt")
                        with open(text_file_path, "w", encoding="utf-8") as txt_file:
                            txt_file.write(extracted_text)
                        st.success(f"Extracted text saved to '{text_file_path}' for student {admin_name} (Matric: {admin_matric}).")
                    else:
                        st.error("Please save the student details before saving the extracted text.")

    if page == "Manage Students" and st.session_state.user_role == "Admin":
        st.subheader("Manage Students")
        all_students = execute_query("SELECT matric_number, name FROM students", fetchall=True)
        if all_students:
            selected_student = st.selectbox("Select a student to view details:", [f"{s[1]} ({s[0]})" for s in all_students], key="select_student")
            if selected_student:
                matric_number = selected_student.split('(')[-1].strip(')')
                student = execute_query("SELECT * FROM students WHERE matric_number = ?", (matric_number,), fetchone=True)
                if student:
                    st.write(f"Matric Number: {student[0]}")
                    st.write(f"Name: {student[1]}")
                    st.write(f"Folder: {student[2]}")

                    # Display the student's captured picture if it exists
                    captured_image_path = os.path.join(student[2], "captured_face.jpg")
                    if os.path.exists(captured_image_path):
                        st.image(captured_image_path, caption="Captured Face", width=200)
                    else:
                        st.info("No captured face image found for this student.")

                    documents = [f for f in os.listdir(student[2]) if f.endswith(('.txt', '.pdf', '.jpg', '.jpeg', '.png'))]
                    if documents:
                        st.subheader("Documents")
                        selected_doc = st.selectbox("Select a document to view:", documents, key="select_doc")
                        if selected_doc:
                            doc_path = os.path.join(student[2], selected_doc)
                            
                            # AI Preview Feature
                            if st.button("Preview Document with AI", key="preview_ai_button"):
                                st.subheader("AI-Extracted Content:")
                                ai_preview = chat_with_ai("Extract detailed text from this document:", file_path=doc_path)
                                st.write(ai_preview)  # Display the AI-extracted content as non-editable text

                            # Option to delete the document
                            if st.button("Delete Document", key="delete_doc_button"):
                                if os.path.exists(doc_path):  # Check if the file exists
                                    try:
                                        os.remove(doc_path)
                                        st.success(f"Document '{selected_doc}' has been deleted.")
                                        # Update the documents list after successful deletion
                                        documents = [f for f in os.listdir(student[2]) if f.endswith(('.txt', '.pdf', '.jpg', '.jpeg', '.png'))]
                                    except Exception as e:
                                        st.error(f"Error deleting document: {e}")
                                else:
                                    st.error(f"Document '{selected_doc}' does not exist.")

                            # Option to download the document
                            st.download_button("Download Document", open(doc_path, "rb").read(), file_name=selected_doc)
                    else:
                        st.info("No documents found for this student.")
        else:
            st.info("No students found.")

    if page == "Analytics Dashboard" and st.session_state.user_role == "Admin":
        st.subheader("Analytics Dashboard")
        total_students = len(execute_query("SELECT matric_number FROM students", fetchall=True))
        total_documents = execute_query("SELECT COUNT(*) FROM document_versions", fetchone=True)[0]
        total_logins = execute_query("SELECT COUNT(*) FROM activity_logs WHERE action LIKE '%Login%'", fetchone=True)[0]
        
        st.write(f"Total Students: {total_students}")
        st.write(f"Total Documents: {total_documents}")
        st.write(f"Total Logins: {total_logins}")

    if page == "Student Panel" and st.session_state.user_role == "Student":
        st.subheader("Student Panel: Retrieve Information")
        if st.session_state.verified_student:
            matric, name = st.session_state.verified_student
            student = execute_query("SELECT * FROM students WHERE matric_number = ?", (matric,), fetchone=True)
            if student:
                # Perform facial recognition scan
                folder = student[2]
                stored_image_path = os.path.join(folder, "captured_face.jpg")
                if os.path.exists(stored_image_path):
                    if not capture_and_verify_face(folder, stored_image_path):
                        st.error("Facial recognition failed. Please try again.")
                        return  # Stop further execution but allow retry
                else:
                    st.error("No stored face image found. Please contact the admin.")
                    return  # Stop further execution if no stored face image is found

                # If facial recognition succeeds, display student data
                st.success(f"Welcome, {student[1]}!")
                documents = [f for f in os.listdir(student[2]) if f.endswith(('.txt', '.pdf', '.jpg', '.jpeg', '.png'))]
                if documents:
                    st.subheader("Your Documents")
                    selected_doc = st.selectbox("Select a document to view:", documents, key="select_doc")
                    if selected_doc:
                        doc_path = os.path.join(student[2], selected_doc)
                        if selected_doc.endswith('.txt'):
                            with open(doc_path, "r", encoding="utf-8") as file:
                                st.text_area(f"Content of {selected_doc}", file.read(), height=300)
                        st.download_button("Download Document", open(doc_path, "rb").read(), file_name=selected_doc)
                else:
                    st.info("No documents found for your account.")
            else:
                st.error("Student record not found. Please contact the admin.")
        else:
            st.error("Please log in to view your information.")

    if page == "AI Study Helper" and st.session_state.user_role == "Student":
        st.subheader("AI Study Helper")
        
        # File upload for AI processing
        uploaded_file = st.file_uploader("Upload a document (PDF, JPG, PNG) for AI to analyze:", type=["pdf", "jpg", "jpeg", "png"], key="ai_file_upload")
        file_path = None
        if uploaded_file:
            file_path = os.path.join("ai_uploaded_files", uploaded_file.name)
            os.makedirs("ai_uploaded_files", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # Query input for AI
        query = st.text_area("Ask a question to the AI (related to the uploaded document or general):", key="ai_query")
        if st.button("Ask AI", key="ask_ai_button"):  # Updated button text
            if query.strip():
                response = chat_with_ai(query, file_path=file_path)  # Pass file path to chat_with_ai
                st.subheader("AI Response:")
                st.write(response)  # Display the AI response as non-editable text
            else:
                st.error("Please enter a question before asking the AI.")

if __name__ == '__main__':
    streamlit_frontend()
