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
from docx import Document  # Ensure you install python-docx: pip install python-docx

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
                                academic_session TEXT)''')
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
    query = "SELECT matric_number, name, folder, academic_session FROM students WHERE matric_number = ?"
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

    # Ensure matric_number is at least 3 characters long
    if len(matric_number) < 3:
        raise ValueError("Matric number must be at least 3 characters long.")
    folder = os.path.join("students_data", f"{name}_{matric_number[-3:]}")  # Student-specific folder
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)
    file.save(file_path)

    text = ""
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = Image.open(file_path)
        text = extract_text_from_image(image)

    text_file_path = os.path.join(folder, f"{file.filename.rsplit('.', 1)[0]}.txt")
    with open(text_file_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text)

    # Save extracted text as a Word document in the student's folder
    word_file_path = os.path.join(folder, f"{file.filename.rsplit('.', 1)[0]}.docx")
    save_text_as_word(text, word_file_path)

    save_document_version(matric_number, file.filename, file_path, text_file_path)
    return jsonify({
        "message": "Document uploaded successfully",
        "file_path": file_path,
        "text_file_path": text_file_path,
        "word_file_path": word_file_path
    })

@app.route('/save_extracted_text', methods=['POST'])
def save_extracted_text():
    """Save extracted text as a .txt file."""
    data = request.json
    matric_number = data.get('matric_number')
    document_name = data.get('document_name')
    extracted_text = data.get('extracted_text')

    if not matric_number or not document_name or not extracted_text:
        return jsonify({"error": "Missing required fields"}), 400

    # Ensure the folder is specific to the student
    folder = os.path.join("students_data", f"{matric_number[-3:]}")
    os.makedirs(folder, exist_ok=True)

    # Save the extracted text as a .txt file
    text_file_path = os.path.join(folder, f"{document_name.rsplit('.', 1)[0]}_extracted.txt")
    try:
        with open(text_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)
        return jsonify({"message": "Extracted text saved successfully", "text_file_path": text_file_path})
    except Exception as e:
        return jsonify({"error": f"Failed to save extracted text: {str(e)}"}), 500

def save_text_as_word(text, word_file_path):
    """Save extracted text as a Word document."""
    doc = Document()
    doc.add_paragraph(text)
    doc.save(word_file_path)

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
    try:
        data = request.json
        username = data.get('username')  # Admin username
        password = data.get('password')  # Admin password
        role = data.get('role')

        if role == "Admin":
            # Admin login logic
            query = "SELECT username, password, role FROM users WHERE username = ?"
            user = execute_query(query, (username,), fetchone=True)
            if user and user[1] == password and user[2] == role:
                return jsonify({"message": f"Welcome, {username}!", "role": "Admin"})
            else:
                return jsonify({"error": "Invalid credentials or role mismatch"}), 401
        else:
            return jsonify({"error": "Only admins are allowed to log in"}), 403

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def verify_admin_face(username):
    """Verify admin's face using facial recognition."""
    st.subheader("Facial Recognition Verification")
    admin_folder = "admin_data"
    stored_encoding_path = os.path.join(admin_folder, f"{username}_face_encoding.npy")

    if not os.path.exists(stored_encoding_path):
        st.error("No stored face encoding found for this admin. Please contact the system administrator.")
        return False

    # Capture face using Streamlit's camera input
    img_file = st.camera_input("Capture your face for verification")

    if img_file:
        try:
            # Save the captured image temporarily
            temp_image_path = os.path.join(admin_folder, f"{username}_temp_captured_face.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(img_file.getvalue())

            # Load the captured image
            captured_image = face_recognition.load_image_file(temp_image_path)

            # Detect face encodings in the captured image
            captured_encodings = face_recognition.face_encodings(captured_image)

            if len(captured_encodings) > 0:
                captured_encoding = captured_encodings[0]

                # Load the stored face encoding
                stored_encoding = np.load(stored_encoding_path)

                # Compare the captured encoding with the stored encoding
                matches = face_recognition.compare_faces([stored_encoding], captured_encoding)
                face_distance = face_recognition.face_distance([stored_encoding], captured_encoding)

                if matches[0]:
                    st.success("Face verified successfully!")
                    return True
                else:
                    st.error("Facial verification failed. Access denied.")
            else:
                st.error("No face detected in the captured image. Please ensure your face is clearly visible and try again.")
        except Exception as e:
            st.error(f"An error occurred during face verification: {e}")
        finally:
            # Clean up the temporary image file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    else:
        st.info("Please capture your face for verification.")
    return False

def get_student_folder(name, matric_number):
    """Generate the folder path for a student."""
    return os.path.join("students_data", f"{name}_{matric_number[-3:]}")

def handle_document_actions(documents, folder):
    """Handle document-related actions like preview, delete, and download."""
    if documents:
        st.subheader("Documents")
        selected_doc = st.selectbox("Select a document to view:", documents, key="select_doc")
        if selected_doc:
            doc_path = os.path.join(folder, selected_doc)

            # Preview Document
            if st.button("Preview Document with AI", key="preview_ai_button"):
                st.subheader("AI-Extracted Content:")
                ai_preview = chat_with_ai("Extract detailed text from this document:", file_path=doc_path)
                st.write(ai_preview)

            # Delete Document
            if st.button("Delete Document", key="delete_doc_button"):
                try:
                    os.remove(doc_path)
                    st.success(f"Document '{selected_doc}' has been deleted.")
                    documents.remove(selected_doc)  # Update the list after deletion
                except Exception as e:
                    st.error(f"Error deleting document: {e}")

            # Download Document
            if os.path.exists(doc_path):
                st.download_button("Download Document", open(doc_path, "rb").read(), file_name=selected_doc)
    else:
        st.info("No documents found for this student.")

def update_student_details(student, updated_name, updated_matric_number, updated_session):
    """Update student details and handle folder renaming."""
    new_folder = get_student_folder(updated_name, updated_matric_number)
    os.makedirs(new_folder, exist_ok=True)

    # Move files if folder name changes
    if student[2] != new_folder:
        for file_name in os.listdir(student[2]):
            os.rename(os.path.join(student[2], file_name), os.path.join(new_folder, file_name))
        if not os.listdir(student[2]):
            os.rmdir(student[2])

    # Update database
    query = '''UPDATE students SET matric_number = ?, name = ?, academic_session = ?, folder = ? WHERE matric_number = ?'''
    execute_query(query, (updated_matric_number, updated_name, updated_session, new_folder, student[0]))
    st.success(f"Student details updated for {updated_name} (Matric: {updated_matric_number}).")

def streamlit_frontend():
    """Streamlit frontend for interacting with the backend."""
    st.title("YABA COLLEGE OF TECHNOLOGY COMPUTER ENGINEERING DEPARTMENT")

    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_role" not in st.session_state:
        st.session_state.user_role = None

    # Get current page from query parameters
    default_page = "Login"
    page = st.query_params.get("page", default_page)

    # Validate page based on login state
    if not st.session_state.logged_in:
        available_pages = ["Login", "Sign Up", "Delete Admin"]
        if page not in available_pages:
            page = default_page
            st.query_params["page"] = page
    else:
        available_pages = ["Admin Panel", "Manage Students", "Analytics Dashboard"]
        if page not in available_pages:
            page = "Admin Panel"
            st.query_params["page"] = page

    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox(
        "Select an option:",
        available_pages,
        index=available_pages.index(page),
        key="nav_select"
    )

    # Update query params if selection changes
    if selected_page != page:
        st.query_params["page"] = selected_page
        st.rerun()

    # Login Page
    if page == "Login":
        st.subheader("Admin Login")
        username = st.text_input("Username:", key="username_input")
        password = st.text_input("Password:", type="password", key="password_input")
        
        if st.button("Login", key="login_button"):
            user = execute_query("SELECT username, password, role FROM users WHERE username = ?", 
                                (username,), fetchone=True)
            if user and user[1] == password and user[2] == "Admin":
                st.success("Login successful! Redirecting to Admin Panel...")
                st.session_state.update({
                    "logged_in": True,
                    "user_role": "Admin",
                    "username": username
                })
                st.query_params["page"] = "Admin Panel"  # Updated query params
                execute_query("INSERT INTO activity_logs (user, action, timestamp) VALUES (?, ?, ?)", 
                            (username, "Admin Login", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                st.rerun()
            else:
                st.error("Invalid credentials or role mismatch.")
	
	# Sign Up Page (keep your existing implementation)
    if page == "Sign Up":
        st.subheader("Admin Sign Up")
        new_username = st.text_input("New Admin Username:", key="new_username_input")
        new_password = st.text_input("New Admin Password:", type="password", key="new_password_input")
        confirm_password = st.text_input("Confirm Password:", type="password", key="confirm_password_input")

        # Capture face for new admin
        st.subheader("Capture Face for Verification")
        img_file = st.camera_input("Capture your face for future verification")

        if st.button("Sign Up", key="signup_button"):
            if not new_username or not new_password or not confirm_password or not img_file:
                st.error("Please fill in all fields and capture your face.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                # Check if the username already exists
                existing_user = execute_query("SELECT username FROM users WHERE username = ?", (new_username,), fetchone=True)
                if existing_user:
                    st.error("Username already exists. Please choose a different username.")
                else:
                    try:
                        # Save face encoding
                        admin_folder = "admin_data"
                        os.makedirs(admin_folder, exist_ok=True)
                        temp_image_path = os.path.join(admin_folder, f"{new_username}_temp_captured_face.jpg")
                        with open(temp_image_path, "wb") as f:
                            f.write(img_file.getvalue())

                        captured_image = face_recognition.load_image_file(temp_image_path)
                        encodings = face_recognition.face_encodings(captured_image)

                        if len(encodings) > 0:
                            stored_encoding_path = os.path.join(admin_folder, f"{new_username}_face_encoding.npy")
                            np.save(stored_encoding_path, encodings[0])
                            os.remove(temp_image_path)  # Clean up temporary image

                            # Create the new admin account
                            query = "INSERT INTO users (username, password, role) VALUES (?, ?, ?)"
                            execute_query(query, (new_username, new_password, "Admin"))
                            st.success("Admin account created successfully! You can now log in.")
                        else:
                            st.error("No face detected in the captured image. Please try again.")
                    except Exception as e:
                        st.error(f"An error occurred while processing the face image: {e}")
						
    if page == "Delete Admin":
        st.subheader("Delete Admin Account")
        admin_username = st.text_input("Enter Admin Username to Delete:", key="delete_admin_username")
        admin_password = st.text_input("Enter Admin Password for Verification:", type="password", key="delete_admin_password")

        if st.button("Delete Admin", key="delete_admin_button"):
            if not admin_username or not admin_password:
                st.error("Please fill in all fields.")
            else:
                # Verify admin credentials
                query = "SELECT username, password FROM users WHERE username = ? AND password = ? AND role = 'Admin'"
                admin = execute_query(query, (admin_username, admin_password), fetchone=True)
                if admin:
                    try:
                        # Delete admin face encoding if it exists
                        admin_folder = "admin_data"
                        face_encoding_path = os.path.join(admin_folder, f"{admin_username}_face_encoding.npy")
                        if os.path.exists(face_encoding_path):
                            os.remove(face_encoding_path)

                        # Delete admin account from the database
                        delete_query = "DELETE FROM users WHERE username = ?"
                        execute_query(delete_query, (admin_username,))
                        st.success(f"Admin account '{admin_username}' has been deleted successfully.")
                    except Exception as e:
                        st.error(f"An error occurred while deleting the admin account: {e}")
                else:
                    st.error("Invalid admin credentials. Please try again.")
					
	 # Logout Handling
    if st.session_state.logged_in:
        if st.sidebar.button("Log Out", key="logout_sidebar"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.query_params.clear()  # Clear all query parameters
            st.rerun()

    if page == "Admin Panel" and st.session_state.user_role == "Admin":
        st.subheader("Admin Panel: Manage Students and Process Documents with AI")

        # Perform facial recognition scan before granting access
        if not verify_admin_face(st.session_state.get("username")):
            st.warning("Facial recognition required to access this panel.")
            return

        # Section to save student details
        st.subheader("Save Student Details")
        admin_matric = st.text_input("Enter Student Matric Number:", key="admin_matric").strip()
        admin_name = st.text_input("Enter Student Name:", key="admin_name")
        admin_session = st.text_input("Enter Academic Session:", key="admin_session")

        if st.button("Save Student Details", key="save_student_details_button"):
            if admin_matric and admin_name and admin_session:
                folder = os.path.join("students_data", f"{admin_name}_{admin_matric[-3:]}")  # Ensure folder name is correct
                os.makedirs(folder, exist_ok=True)
                query = '''INSERT OR REPLACE INTO students (matric_number, name, folder, academic_session) 
                           VALUES (?, ?, ?, ?)'''
                execute_query(query, (admin_matric, admin_name, folder, admin_session))
                st.success(f"Student details saved for {admin_name} (Matric: {admin_matric}).")
            else:
                st.error("Please fill in all the fields.")

        # Section for AI-assisted document upload and processing
        st.subheader("Upload and Process Documents with AI")
        admin_doc_file = st.file_uploader("Upload a document (PDF, JPG, PNG) for AI-assisted text extraction:", type=["pdf", "jpg", "jpeg", "png"], key="admin_doc_file")
        if admin_doc_file:
            folder = os.path.join("students_data", f"{admin_name}_{admin_matric[-3:]}")  # Student-specific folder
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, admin_doc_file.name)
            
            # Save the original document
            if st.button("Save Original Document", key="save_original_doc_button"):
                with open(file_path, "wb") as f:
                    f.write(admin_doc_file.read())
                st.success(f"Original document '{admin_doc_file.name}' saved successfully!")

            # Extract text using AI
            if st.button("Extract Text with AI", key="extract_text_ai_button"):
                extracted_text = ""
                if admin_doc_file.name.endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(file_path)
                elif admin_doc_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image = Image.open(file_path)
                    extracted_text = extract_text_from_image(image)

                st.subheader("AI-Extracted Text:")
                st.write(extracted_text)  # Display the extracted text as non-editable content

                # Save the extracted text as a .txt file
                if st.button("Save Extracted Text as Text", key="save_extracted_text_txt_button"):
                    response = requests.post(
                        "http://127.0.0.1:5000/save_extracted_text",
                        json={
                            "matric_number": admin_matric,
                            "document_name": admin_doc_file.name,
                            "extracted_text": extracted_text
                        }
                    )
                    if response.status_code == 200:
                        st.success(f"Extracted text saved successfully: {response.json().get('text_file_path')}")
                    else:
                        st.error(f"Failed to save extracted text: {response.json().get('error')}")

    if page == "Manage Students" and st.session_state.user_role == "Admin":
        st.subheader("Manage Students")

        # Perform facial recognition scan before granting access
        if not verify_admin_face(st.session_state.get("username")):
            st.warning("Facial recognition required to access this panel.")
            return

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

                    if not os.path.exists(student[2]):
                        st.warning(f"The folder '{student[2]}' does not exist.")
                        return

                    # Ensure the folder exists before listing its contents
                    documents = [f for f in os.listdir(student[2]) if f.endswith(('.txt', '.pdf', '.jpg', '.jpeg', '.png'))]

                    # Handle document actions
                    handle_document_actions(documents, student[2])

                    # Update student details
                    st.subheader("Update Student Details")
                    updated_name = st.text_input("Update Name:", value=student[1], key="update_name")
                    updated_session = st.text_input("Update Academic Session:", value=student[3], key="update_session")
                    updated_matric_number = st.text_input("Update Matric Number:", value=student[0], key="update_matric_number")

                    # Update face image
                    st.subheader("Update Face Image")
                    face_image_file = st.camera_input("Capture or upload a new face image for the student", key="update_face_image")
                    if face_image_file:
                        face_image_path = os.path.join(student[2], "captured_face.png")
                        with open(face_image_path, "wb") as f:
                            f.write(face_image_file.getvalue())
                        st.success("Face image updated successfully!")

                    # Upload new document
                    st.subheader("Upload New Document")
                    new_document = st.file_uploader("Upload a document (PDF, JPG, PNG):", type=["pdf", "jpg", "jpeg", "png"], key="upload_new_document")
                    if new_document:
                        document_path = os.path.join(student[2], new_document.name)
                        with open(document_path, "wb") as f:
                            f.write(new_document.read())
                        st.success(f"Document '{new_document.name}' uploaded successfully!")

                        if st.button("Extract Text from New Document", key="extract_text_new_doc_button"):
                            extracted_text = ""
                            if new_document.name.endswith(".pdf"):
                                extracted_text = extract_text_from_pdf(document_path)
                            elif new_document.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                image = Image.open(document_path)
                                extracted_text = extract_text_from_image(image)

                            st.subheader("Extracted Text:")
                            st.write(extracted_text)

                            if st.button("Save Extracted Text as Text", key="save_extracted_text_new_doc_button"):
                                text_file_path = os.path.join(student[2], f"{new_document.name.rsplit('.', 1)[0]}_extracted.txt")
                                with open(text_file_path, "w", encoding="utf-8") as txt_file:
                                    txt_file.write(extracted_text)
                                st.success(f"Extracted text saved as text file to '{text_file_path}'!")

                    # Save updated details
                    if st.button("Save Updates", key="save_updates_button"):
                        if updated_name and updated_matric_number and updated_session:
                            update_student_details(student, updated_name, updated_matric_number, updated_session)
                        else:
                            st.error("Please fill in all the fields.")
                else:
                    st.info("No students found.")
            else:
                st.info("No students found.")
        else:
            st.info("No students found.")

    if page == "Analytics Dashboard" and st.session_state.user_role == "Admin":
        st.subheader("Analytics Dashboard")

        # Perform facial recognition scan before granting access
        if not verify_admin_face(st.session_state.get("username")):
            st.warning("Facial recognition required to access this panel.")
            return

        # Fetch analytics data
        total_students = len(execute_query("SELECT matric_number FROM students", fetchall=True))
        total_documents = execute_query("SELECT COUNT(*) FROM document_versions", fetchone=True)[0]
        total_logins = execute_query("SELECT COUNT(*) FROM activity_logs WHERE action LIKE '%Login%'", fetchone=True)[0]

        # Display data as charts
        st.write("### Overview")
        st.bar_chart({
            "Metrics": ["Total Students", "Total Documents", "Total Logins"],
            "Counts": [total_students, total_documents, total_logins]
        })

        # Additional analytics: Logins over time
        st.write("### Logins Over Time")
        login_data = execute_query("SELECT timestamp FROM activity_logs WHERE action LIKE '%Login%'", fetchall=True)
        if login_data:
            login_timestamps = [datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") for row in login_data]
            login_counts = {date.strftime("%Y-%m-%d"): 0 for date in login_timestamps}
            for timestamp in login_timestamps:
                login_counts[timestamp.strftime("%Y-%m-%d")] += 1

            st.line_chart({
                "Date": list(login_counts.keys()),
                "Logins": list(login_counts.values())
            })
        else:
            st.info("No login data available.")

if __name__ == '__main__':
    streamlit_frontend()