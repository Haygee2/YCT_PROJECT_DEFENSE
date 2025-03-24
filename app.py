import os
import sqlite3
import numpy as np
import pytesseract
import streamlit as st
from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time  # Add this import for the delay
from dotenv import load_dotenv  # Add this import
from chatbot import chat_with_ai  # Add this import
import asyncio  # Add this import

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Debugging: Print the current working directory and list files
print(f"Current working directory: {os.getcwd()}")
print(f"Files in the current directory: {os.listdir(os.getcwd())}")

# Verify if the environment variable is loaded
api_key = os.getenv('OPENROUTER_API_KEY')
print(f"OPENROUTER_API_KEY: {api_key}")

# Check if the API key is loaded correctly
if not api_key:
    print("Error: OPENROUTER_API_KEY not found in environment variables.")
else:
    print("OPENROUTER_API_KEY loaded successfully.")

# Set up OpenAI API (replace with your API key)
OPENROUTER_API_KEY = api_key
if not OPENROUTER_API_KEY:
    st.error("API key not found. Please set the OPENROUTER_API_KEY in the .env file.")
else:
    try:
        client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=OPENROUTER_API_KEY
        )
        print("OpenAI client initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        print(f"Failed to initialize OpenAI client: {e}")

# Set the path to Tesseract-OCR executable
if os.name == 'nt':  # Windows
    tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    if not os.path.exists(tesseract_cmd):
        st.error("Tesseract executable not found at the specified path. Please install Tesseract OCR and ensure the path is correct.")
        raise FileNotFoundError("Tesseract executable not found.")
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    os.environ["TESSDATA_PREFIX"] = r"C:\\Program Files\\Tesseract-OCR\\tessdata"
else:  # Unix/Linux/Mac
    # Use default system path or set accordingly
    tesseract_cmd = "tesseract"
    if not shutil.which(tesseract_cmd):
        st.error("Tesseract is not installed or it's not in your PATH. Please install Tesseract OCR and try again.")
        raise FileNotFoundError("Tesseract executable not found.")

# Admin authentication (hardcoded for now)
ADMINS = {"admin": "password"}  # Change credentials as needed

# Student authentication (hardcoded for now)
STUDENTS = {"student1": "password1"}  # Change credentials as needed

# Database setup
DB_PATH = "students.db"
FACE_ENCODINGS_DIR = "face_encodings"
os.makedirs(FACE_ENCODINGS_DIR, exist_ok=True)

def init_db():
    """Initialize the database and create the students table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                            matric_number TEXT PRIMARY KEY,
                            name TEXT,
                            folder TEXT,
                            face_image TEXT,
                            face_encoding_path TEXT,
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
                            timestamp TEXT)''')
        
        # Migration logic to add the email column if it doesn't exist
        cursor.execute("PRAGMA table_info(students)")
        columns = [column[1] for column in cursor.fetchall()]
        if "email" not in columns:
            cursor.execute("ALTER TABLE students ADD COLUMN email TEXT")
            conn.commit()

# Call the init_db function to ensure the database is initialized
init_db()

def execute_query(query, params=(), fetchone=False, fetchall=False):
    """
    General function to execute queries safely.
    """
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
        st.error(f"Database error: {e}")
        print(f"Database error: {e}")  # Improved logging
        return None

def get_student_info(matric_number):
    """Fetch student details by matric_number."""
    return execute_query("SELECT matric_number, name, folder, face_image, face_encoding_path, email FROM students WHERE matric_number = ?", (matric_number,), fetchone=True)

def get_all_students():
    """Fetch all students from the database."""
    return execute_query("SELECT matric_number, name FROM students", fetchall=True)

def store_student(matric_number, name, folder, face_image_path="", face_encoding_path="", email=""):
    """Stores or updates student information in the database."""
    query = '''INSERT OR REPLACE INTO students (matric_number, name, folder, face_image, face_encoding_path, email) 
               VALUES (?, ?, ?, ?, ?, ?)'''
    execute_query(query, (matric_number, name, folder, face_image_path, face_encoding_path, email))

def get_student_folder(matric_number, name):
    """Retrieve or create the student's folder path."""
    student = get_student_info(matric_number)
    
    if student:
        return student[2]  # Folder path from database
    else:
        folder_path = os.path.join("students_data", f"{name}_{matric_number[-3:]}")
        os.makedirs(folder_path, exist_ok=True)
        store_student(matric_number, name, folder_path)  # Store in DB
        return folder_path

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract is not installed or it's not in your PATH. Please install Tesseract OCR and try again.")
        return "OCR Error: Tesseract is not installed or it's not in your PATH."
    except Exception as e:
        return f"OCR Error: {e}"

def pdf_to_images(pdf_path):
    """Convert PDF pages to images."""
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    
    return images

def save_document(folder, file, matric_number):
    """Save uploaded documents (PDF or image) and extract text if necessary."""
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.name)

    file_content = file.read()  # Read file content into memory

    with open(file_path, "wb") as f:
        f.write(file_content)

    # Extract text if it's a PDF or image
    text = ""
    if file.name.endswith(".pdf"):
        with st.spinner('Processing PDF document...'):
            images = pdf_to_images(file_path)
            text = "".join(extract_text_from_image(img) for img in images)
    elif file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = Image.open(file_path)
        text = extract_text_from_image(image)

    # Save extracted text to a .txt file
    text_file_path = file_path.rsplit('.', 1)[0] + ".txt"
    with open(text_file_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text)

    # Save document version
    save_document_version(matric_number, file.name, text_file_path)

    return text_file_path

def save_document_version(matric_number, document_name, file_path):
    """Save a new version of the document."""
    version = get_latest_document_version(matric_number, document_name) + 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = '''INSERT INTO document_versions (matric_number, document_name, version, file_path, timestamp)
               VALUES (?, ?, ?, ?, ?)'''
    execute_query(query, (matric_number, document_name, version, file_path, timestamp))

def get_latest_document_version(matric_number, document_name):
    """Get the latest version number of a document."""
    query = '''SELECT MAX(version) FROM document_versions WHERE matric_number = ? AND document_name = ?'''
    result = execute_query(query, (matric_number, document_name), fetchone=True)
    return result[0] if result[0] else 0

def capture_face(camera_index=0, student_folder=""):
    cap = cv2.VideoCapture(camera_index)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_captured = False
    captured_image_path = os.path.join(student_folder, "captured_face.jpg")
    
    st.info("Get ready! Capturing the image in 5 seconds...")
    time.sleep(5)  # Add a 5-second delay
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            st.error("Failed to capture image from camera. Please ensure the camera is connected and accessible.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imshow('Face Capture', frame)
        
        if len(faces) > 0:
            cv2.imwrite(captured_image_path, frame)
            face_captured = True
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if face_captured:
        st.success("Face captured successfully!")
        st.image(captured_image_path, caption="Captured Face", width=700)  # Show the captured face
    else:
        st.error("Failed to capture face.")

    print("Face capture process completed.")

def verify_matric_number(matric_number):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE matric_number = ?", (matric_number,))
    student_data = cursor.fetchone()
    conn.close()
    
    if student_data:
        print("Matric number verified!")
        return student_data
    else:
        print("Matric number not found!")
        return None

def ai_prompt_page():
    """AI Study Helper Page."""
    st.title("AI Study Helper")
    
    # Sidebar for navigation within the AI Study Helper
    st.sidebar.title("AI Study Helper Navigation")
    page = st.sidebar.radio("Go to", ["Ask a Question"], key="ai_nav")  # Removed "Previous Queries"
    
    if page == "Ask a Question":
        st.header("Ask a Question")
        query = st.text_area("What can I help you with today:", height=150, key="query_input", on_change=submit_query)  # Increased height and added on_change
        if st.button("Submit") or st.session_state.get("submit_query", False):
            response = chat_with_ai(query)
            st.text_area("AI Response:", value=response, height=350, key="response_output")  # Increased height

def submit_query():
    st.session_state.submit_query = True

def list_student_documents(folder):
    """List all text documents in the student's folder."""
    if not os.path.exists(folder):
        return []
    
    documents = []
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            documents.append(file)
    
    return documents

def update_student_info(matric_number, name, folder, face_image_path="", face_encoding_path="", email=""):
    """Update student information in the database."""
    query = '''UPDATE students SET name = ?, folder = ?, face_image = ?, face_encoding_path = ?, email = ?
               WHERE matric_number = ?'''
    execute_query(query, (name, folder, face_image_path, face_encoding_path, email, matric_number))

def log_activity(user, action):
    """Log user activity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = '''INSERT INTO activity_logs (user, action, timestamp) VALUES (?, ?, ?)'''
    execute_query(query, (user, action, timestamp))

def send_email_notification(to_email, subject, body):
    """Send an email notification."""
    from_email = "your_email@example.com"
    from_password = "your_app_password"  # Use an app password if 2-Step Verification is enabled

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def main():
    # Ensure the event loop is properly initialized
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    st.title("YABA COLLEGE OF TECHNOLOGY COMPUTER ENGINEERING DEPARTMENT")

    # Sidebar for navigation and logout
    st.sidebar.title("Navigation")
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "verified_student" not in st.session_state:
        st.session_state.verified_student = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None

    if st.session_state.logged_in:
        if st.sidebar.button("Log Out", key="logout_sidebar"):
            st.session_state.logged_in = False
            st.session_state.verified_student = None
            st.session_state.user_role = None
            st.experimental_set_query_params(logged_in=False)

    if not st.session_state.logged_in:
        st.subheader("Login")
        role = st.radio("Select Role:", ["Admin", "Student"], key="role_radio")
        username = st.text_input("Username:", key="username_input")
        password = st.text_input("Password:", type="password", key="password_input", on_change=submit_login)  # Added on_change
        
        if st.button("Login", key="login_button") or st.session_state.get("submit_login", False):
            if role == "Admin" and username in ADMINS and ADMINS[username] == password:
                st.session_state.logged_in = True
                st.session_state.user_role = "Admin"
                st.success("Admin Login Successful!")
                st.experimental_set_query_params(logged_in=True)
                log_activity(username, "Admin Login")
            elif role == "Student" and username in STUDENTS and STUDENTS[username] == password:
                st.session_state.logged_in = True
                st.session_state.user_role = "Student"
                st.success("Student Login Successful!")
                st.experimental_set_query_params(logged_in=True)
                log_activity(username, "Student Login")
                st.session_state.verified_student = (username, STUDENTS[username])  # Set verified student
            else:
                st.error("Invalid Credentials")
        st.stop()  # Prevents execution of any further code until login is done

    # Navigation
    if st.session_state.user_role == "Admin":
        page = st.sidebar.radio("Go to", ["Admin Panel", "Manage Students", "Analytics Dashboard"], key="admin_nav")
    else:
        page = st.sidebar.radio("Go to", ["Student Panel", "AI Study Helper"], key="student_nav")

    if page == "Admin Panel" and st.session_state.user_role == "Admin":
        st.subheader("Admin Panel: Upload Student Documents")
        admin_matric = st.text_input("Enter Student Matric Number:", key="admin_matric").strip()
        admin_name = st.text_input("Enter Student Name:", key="admin_name")
        doc_file = st.file_uploader("Upload Document (Image or PDF)", type=["jpg", "png", "pdf"], key="doc_file")
        admin_email = st.text_input("Enter Student Email (for notifications):", key="admin_email")
        camera_index = st.number_input("Enter Camera Index (0 for built-in, 1 for external, etc.):", min_value=0, value=0, step=1, key="camera_index_admin")

        if st.button("Save Document", key="save_doc_button"):
            if admin_matric and admin_name and doc_file:
                with st.spinner('Saving document...'):
                    folder = get_student_folder(admin_matric, admin_name)
                    save_document(folder, doc_file, admin_matric)
                    st.success(f"Data saved for {admin_name} (Matric: {admin_matric}).")
                    log_activity(st.session_state.user_role, f"Uploaded document for {admin_matric}")
                    if admin_email:
                        send_email_notification(admin_email, "New Document Uploaded", f"A new document has been uploaded for {admin_name} (Matric: {admin_matric}).")
            else:
                st.error("Please enter all required details.")
        
        if st.button("Capture Student", key="capture_student_button"):
            if admin_matric and admin_name:
                folder = get_student_folder(admin_matric, admin_name)
                capture_face(camera_index, folder)
                st.success(f"Face captured for {admin_name} (Matric: {admin_matric}).")
                log_activity(st.session_state.user_role, f"Captured face for {admin_matric}")
            else:
                st.error("Please enter the student's matric number and name.")

    if page == "Manage Students" and st.session_state.user_role == "Admin":
        st.subheader("Manage Students")
        all_students = get_all_students()
        if all_students:
            selected_student = st.selectbox("Select a student to view details:", [f"{s[1]} ({s[0]})" for s in all_students], key="select_student")
            if selected_student:
                matric_number = selected_student.split('(')[-1].strip(')')
                student = get_student_info(matric_number)
                if student:
                    st.write(f"Matric Number: {student[0]}")
                    st.write(f"Name: {student[1]}")
                    st.write(f"Folder: {student[2]}")
                    if student[3] and os.path.exists(student[3]):
                        st.image(student[3], caption="Registered Face", width=200)
                    documents = list_student_documents(student[2])
                    if documents:
                        st.subheader("Documents")
                        for doc in documents:
                            doc_path = os.path.join(student[2], doc)
                            with open(doc_path, "r", encoding="utf-8") as file:
                                st.text_area(f"Content of {doc}", file.read(), height=300)
                    else:
                        st.info("No documents found for this student.")
                    
                    # Form to update student details
                    st.subheader("Update Student Details")
                    new_matric_number = st.text_input("Matric Number:", value=student[0], key="update_matric_number")
                    new_name = st.text_input("Name:", value=student[1], key="update_name")
                    new_email = st.text_input("Email:", value=student[5], key="update_email")
                    
                    if st.button("Update Details", key="update_details_button"):
                        update_student_info(new_matric_number, new_name, student[2], student[3], student[4], new_email)
                        st.success("Student details updated successfully.")
                        log_activity(st.session_state.user_role, f"Updated details for {new_matric_number}")
                        if new_email:
                            send_email_notification(new_email, "Student Information Updated", f"Your information has been updated for {new_name} (Matric: {new_matric_number}).")
                    
                    # Recapture face image
                    camera_index = st.number_input("Enter Camera Index (0 for built-in, 1 for external, etc.):", min_value=0, value=0, step=1, key="camera_index_manage")
                    if st.button("Recapture Face Image", key=f"recapture_face_button_{matric_number}"):
                        capture_face(camera_index, student[2])
                        st.success("Face image recaptured successfully.")
                        log_activity(st.session_state.user_role, f"Recaptured face image for {new_matric_number}")
        else:
            st.info("No students found.")

    if page == "Analytics Dashboard" and st.session_state.user_role == "Admin":
        st.subheader("Analytics Dashboard")
        total_students = len(get_all_students())
        total_documents = execute_query("SELECT COUNT(*) FROM document_versions", fetchone=True)[0]
        total_logins = execute_query("SELECT COUNT(*) FROM activity_logs WHERE action LIKE '%Login%'", fetchone=True)[0]
        
        st.write(f"Total Students: {total_students}")
        st.write(f"Total Documents: {total_documents}")
        st.write(f"Total Logins: {total_logins}")

    if page == "Student Panel" and st.session_state.user_role == "Student":
        st.subheader("Student Panel: Retrieve Information")
        
        # Method selection
        verification_method = st.radio(
            "Select verification method:",
            ["Matric Number", "Facial Recognition"],
            key="verification_method"
        )
        
        if verification_method == "Matric Number":
            student_matric = st.text_input("Enter Matric Number:", key="student_matric").strip()
            
            if st.button("Get Information", key="get_info_button") or st.session_state.get("submit_info", False):
                if student_matric:
                    student = get_student_info(student_matric)
                    if student:
                        st.session_state.verified_student = (student[0], student[1])
                    else:
                        st.error(f"No student found with matric number: {student_matric}")
                else:
                    st.error("Please enter a matric number")
        
        elif verification_method == "Facial Recognition":
            camera_index = st.number_input("Enter Camera Index (0 for built-in, 1 for external, etc.):", min_value=0, value=0, step=1, key="camera_index_student")
            if st.button("Start Face Capture", key="start_face_capture_button"):
                if st.session_state.verified_student:
                    student_folder = get_student_folder(st.session_state.verified_student[0], st.session_state.verified_student[1])
                    capture_face(camera_index, student_folder)
                else:
                    st.error("Please verify your matric number first.")
                # Here you would add the logic to verify the captured face with stored encodings

        # If student is verified, show their information
        if st.session_state.verified_student:
            matric, name = st.session_state.verified_student
            student = get_student_info(matric)
            
            if student:
                st.success(f"Welcome, {student[1]}!")
                st.subheader("Your Information")
                st.write(f"Matric Number: {student[0]}")
                st.write(f"Name: {student[1]}")
                
                # Show student's face if available
                if student[3] and os.path.exists(student[3]):
                    st.image(student[3], caption="Registered Face", width=200)
                
                # List student documents
                documents = list_student_documents(student[2])
                if documents:
                    st.subheader("Your Documents")
                    selected_doc = st.selectbox("Select a document to view:", documents, key="select_doc")
                    
                    if selected_doc:
                        doc_path = os.path.join(student[2], selected_doc)
                        if os.path.exists(doc_path):
                            with open(doc_path, "r", encoding="utf-8") as file:
                                st.text_area(f"Content of {selected_doc}", file.read(), height=300)
                else:
                    st.info("No documents found for your account.")
                
                # Add a clear button to reset verification
                if st.button("Log Out", key="logout_button"):
                    st.session_state.verified_student = None
                    st.experimental_set_query_params(logged_in=False)

    if page == "AI Study Helper" and st.session_state.user_role == "Student":
        ai_prompt_page()

def submit_login():
    st.session_state.submit_login = True

if __name__ == "__main__":
    main()
