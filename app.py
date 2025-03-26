import os
import sqlite3
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
import numpy as np  # Add this import

# Check for face_recognition availability
try:
    import face_recognition  # Add this import
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    st.warning("Facial recognition features are unavailable because the 'face_recognition' library is not installed.")

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Debugging: Print the current working directory and list files
print(f"Current working directory: {os.getcwd()}")
print(f"Files in the current directory: {os.listdir(os.getcwd())}")

# Verify if the environment variable is loaded
api_key = os.getenv('OPENROUTER_API_KEY')
if not api_key:
    st.error("Error: OPENROUTER_API_KEY not found in environment variables. Please check your .env file.")
else:
    st.success("OPENROUTER_API_KEY loaded successfully.")

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
        st.success("OpenAI client initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")

# Set the path to Tesseract-OCR executable
if os.name == 'nt':  # Windows
    tesseract_cmd = os.path.join("C:", "Program Files", "Tesseract-OCR", "tesseract.exe")
    if not os.path.exists(tesseract_cmd):
        st.error("Tesseract executable not found at the specified path. Please install Tesseract OCR and ensure the path is correct.")
        raise FileNotFoundError("Tesseract executable not found.")
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    os.environ["TESSDATA_PREFIX"] = os.path.join("C:", "Program Files", "Tesseract-OCR", "tessdata")

# Verify OpenCV installation
import cv2
print(cv2.__version__)  # Check OpenCV version
print(cv2.getBuildInformation())  # Check if compiled with V4L support

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
                            text_file_path TEXT,
                            timestamp TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            username TEXT PRIMARY KEY,
                            password TEXT,
                            role TEXT)''')  # Add users table for sign-up and login
        
        # Migration logic to add the email column if it doesn't exist
        cursor.execute("PRAGMA table_info(students)")
        columns = [column[1] for column in cursor.fetchall()]
        if "email" not in columns:
            cursor.execute("ALTER TABLE students ADD COLUMN email TEXT")
            conn.commit()
        
        # Migration logic to add the text_file_path column if it doesn't exist
        cursor.execute("PRAGMA table_info(document_versions)")
        columns = [column[1] for column in cursor.fetchall()]
        if "text_file_path" not in columns:
            cursor.execute("ALTER TABLE document_versions ADD COLUMN text_file_path TEXT")
            conn.commit()

# Call the init_db function to ensure the database is initialized
init_db()

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

def get_user(username):
    """Fetch user details by username."""
    return execute_query("SELECT username, password, role FROM users WHERE username = ?", (username,), fetchone=True)

def store_user(username, password, role):
    """Stores a new user in the database."""
    query = '''INSERT INTO users (username, password, role) VALUES (?, ?, ?)'''
    execute_query(query, (username, password, role))

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
        folder_path = student[2]  # Folder path from database
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
        return folder_path
    else:
        folder_path = os.path.join("students_data", f"{name}_{matric_number[-3:]}")
        os.makedirs(folder_path, exist_ok=True)
        store_student(matric_number, name, folder_path)  # Store in DB
        return folder_path

def preprocess_image(image):
    """Preprocess the image for better OCR results."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Resize the image
    scale_percent = 150  # Percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    
    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return Image.fromarray(sharpened)

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        return pytesseract.image_to_string(preprocessed_image)
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract is not installed or it's not in your PATH. Please install Tesseract OCR and try again.")
        return "OCR Error: Tesseract is not installed or it's not in your PATH."
    except Exception as e:
        st.error(f"OCR Error: {e}")
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
    save_document_version(matric_number, file.name, file_path, text_file_path)

    return file_path, text_file_path

def update_document(folder, file, matric_number):
    """Update an existing document (PDF or image) and extract text if necessary."""
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
    save_document_version(matric_number, file.name, file_path, text_file_path)

    return file_path, text_file_path

def get_latest_document_version(matric_number, document_name):
    """Get the latest version number of a document for a student."""
    query = '''SELECT MAX(version) FROM document_versions WHERE matric_number = ? AND document_name = ?'''
    result = execute_query(query, (matric_number, document_name), fetchone=True)
    return result[0] if result[0] is not None else 0

def save_document_version(matric_number, document_name, file_path, text_file_path):
    """Save a new version of the document."""
    version = get_latest_document_version(matric_number, document_name) + 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = '''INSERT INTO document_versions (matric_number, document_name, version, file_path, text_file_path, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)'''
    execute_query(query, (matric_number, document_name, version, file_path, text_file_path, timestamp))

def capture_face_streamlit(student_folder):
    """Capture face using Streamlit's camera input."""
    if not FACE_RECOGNITION_AVAILABLE:
        st.error("Facial recognition is not supported in this environment.")
        return
    st.title("Capture Face Image")
    img_file = st.camera_input("Take a picture")
    if img_file:
        try:
            temp_image_path = os.path.join(student_folder, "temp_captured_face.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(img_file.getvalue())

            captured_image = face_recognition.load_image_file(temp_image_path)
            captured_encodings = face_recognition.face_encodings(captured_image)

            if len(captured_encodings) > 0:
                captured_encoding = captured_encodings[0]
                face_encoding_path = os.path.join(student_folder, "face_encoding.npy")
                np.save(face_encoding_path, captured_encoding)
                st.success("Face captured and encoding saved successfully!")
            else:
                st.error("No face detected. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

def verify_face(student_folder):
    """Verify face using Streamlit's camera input."""
    if not FACE_RECOGNITION_AVAILABLE:
        st.error("Facial recognition is not supported in this environment.")
        return False
    st.title("Facial Recognition Verification")
    img_file = st.camera_input("Capture your face for verification")
    if img_file:
        try:
            temp_image_path = os.path.join(student_folder, "temp_captured_face.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(img_file.getvalue())

            captured_image = face_recognition.load_image_file(temp_image_path)
            captured_encodings = face_recognition.face_encodings(captured_image)

            if len(captured_encodings) > 0:
                captured_encoding = captured_encodings[0]
                face_encoding_path = os.path.join(student_folder, "face_encoding.npy")
                if os.path.exists(face_encoding_path):
                    stored_encoding = np.load(face_encoding_path)
                    matches = face_recognition.compare_faces([stored_encoding], captured_encoding)
                    if matches[0]:
                        st.success("Face verified successfully!")
                        return True
                    else:
                        st.error("Face verification failed. Please try again.")
                else:
                    st.error("No stored face encoding found.")
            else:
                st.error("No face detected. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    return False

def ai_extract_text(image):
    """Extract text from an image using AI."""
    try:
        response = client.Completions.create(
            model="text-davinci-003",
            prompt="Extract the text from this image: " + image,
            max_tokens=1000
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"AI Text Extraction Error: {e}")
        return f"AI Text Extraction Error: {e}"

def ai_prompt_page():
    """AI Study Helper Page."""
    st.title("AI Study Helper")
    
    # Sidebar for navigation within the AI Study Helper
    st.sidebar.title("AI Study Helper Navigation")
    page = st.sidebar.radio("Go to", ["Ask a Question", "Extract Text from Image"], key="ai_nav")
    
    if page == "Ask a Question":
        st.header("Ask a Question")
        query = st.text_area("What can I help you with today:", height=150, key="query_input", on_change=submit_query)
        if st.button("Submit") or st.session_state.get("submit_query", False):
            response = chat_with_ai(query)
            st.text_area("AI Response:", value=response, height=350, key="response_output")
    
    if page == "Extract Text from Image":
        st.header("Extract Text from Image")
        uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png", "pdf"], key="image_file")
        if uploaded_file:
            if uploaded_file.name.endswith(".pdf"):
                images = pdf_to_images(uploaded_file)
                extracted_text = "".join(ai_extract_text(img) for img in images)
            else:
                image = Image.open(uploaded_file)
                extracted_text = ai_extract_text(image)
            
            st.text_area("Extracted Text:", value=extracted_text, height=350, key="extracted_text_output")

def submit_query():
    st.session_state.submit_query = True

def list_student_documents(folder):
    """List all documents for a student."""
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder) if f.endswith(('.txt', '.pdf', '.jpg', '.jpeg', '.png'))]

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

def add_custom_css():
    """Add custom CSS for a modern and interactive design."""
    st.markdown("""
        <style>
        /* Main background and text color */
        .main {
            background-color: #1e1e1e;
            color: #f5f5f5;
        }
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #2b2b2b;
            color: #f5f5f5;
        }
        .sidebar .sidebar-content a {
            color: #f5f5f5;
        }
        .sidebar .sidebar-content a:hover {
            color: #1db954;
        }
        /* Button styling */
        .stButton>button {
            background-color: #1db954;
            color: white;
            border: none;
            border-radius: 25px;
            padding: 10px 20px;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1ed760;
        }
        /* Text input styling */
        .stTextInput>div>div>input {
            border: 2px solid #1db954;
            border-radius: 25px;
            padding: 10px;
            background-color: #2b2b2b;
            color: #f5f5f5;
        }
        .stTextInput>div>div>input:focus {
            border-color: #1ed760;
        }
        /* Radio button styling */
        .stRadio>div>div>label {
            font-size: 16px;
            color: #f5f5f5;
        }
        .stRadio>div>div>div>input:checked+div {
            background-color: #1db954;
            color: white;
        }
        /* Header styling */
        h1, h2, h3, h4, h5, h6 {
            color: #1db954;
        }
        /* Spinner styling */
        .stSpinner>div>div {
            border-top-color: #1db954;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    # Add custom CSS
    add_custom_css()

    # Ensure the event loop is properly initialized
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

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
        password = st.text_input("Password:", type="password", key="password_input", on_change=submit_login)
        
        if st.button("Login", key="login_button") or st.session_state.get("submit_login", False):
            user = get_user(username)
            if user and user[1] == password and user[2] == role:
                st.session_state.logged_in = True
                st.session_state.user_role = role
                st.success(f"{role} Login Successful!")
                st.experimental_set_query_params(logged_in=True)
                log_activity(username, f"{role} Login")
                if role == "Student":
                    st.session_state.verified_student = (username, password)
            else:
                st.error("Invalid Credentials")
        st.stop()

    if page == "Sign Up":
        st.subheader("Sign Up")
        new_username = st.text_input("New Username:", key="new_username_input")
        new_password = st.text_input("New Password:", type="password", key="new_password_input")
        new_role = st.radio("Select Role:", ["Admin", "Student"], key="new_role_radio")
        
        if st.button("Create Account", key="create_account_button"):
            if get_user(new_username):
                st.error("Username already exists. Please choose a different username.")
            else:
                store_user(new_username, new_password, new_role)
                st.success("Account created successfully! You can now log in.")

    if st.session_state.logged_in:
        if st.sidebar.button("Log Out", key="logout_sidebar"):
            st.session_state.logged_in = False
            st.session_state.verified_student = None
            st.session_state.user_role = None
            st.experimental_set_query_params(logged_in=False)

    if page == "Admin Panel" and st.session_state.user_role == "Admin":
        st.subheader("Admin Panel: Upload Student Documents")
        admin_matric = st.text_input("Enter Student Matric Number:", key="admin_matric").strip()
        admin_name = st.text_input("Enter Student Name:", key="admin_name")
        doc_file = st.file_uploader("Upload Document (Image or PDF)", type=["jpg", "png", "pdf"], key="doc_file")
        admin_email = st.text_input("Enter Student Email (for notifications):", key="admin_email")

        if st.button("Save Document", key="save_doc_button"):
            if admin_matric and admin_name and doc_file:
                with st.spinner('Saving document...'):
                    folder = get_student_folder(admin_matric, admin_name)
                    file_path, text_file_path = save_document(folder, doc_file, admin_matric)
                    st.success(f"Data saved for {admin_name} (Matric: {admin_matric}).")
                    log_activity(st.session_state.user_role, f"Uploaded document for {admin_matric}")
                    if admin_email:
                        send_email_notification(admin_email, "New Document Uploaded", f"A new document has been uploaded for {admin_name} (Matric: {admin_matric}).")
            else:
                st.error("Please enter all required details.")
        
        if st.button("Capture Student", key="capture_student_button"):
            if admin_matric and admin_name:
                folder = get_student_folder(admin_matric, admin_name)
                capture_face_streamlit(folder)
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
                        selected_doc = st.selectbox("Select a document to view:", documents, key="select_doc")
                        if selected_doc:
                            doc_path = os.path.join(student[2], selected_doc)
                            if selected_doc.endswith('.txt'):
                                if os.path.exists(doc_path):
                                    if st.button("Show Document", key="show_doc_button"):
                                        with open(doc_path, "r", encoding="utf-8") as file:
                                            st.text_area(f"Content of {selected_doc}", file.read(), height=300)
                            update_doc_file = st.file_uploader("Update Document (Image or PDF)", type=["jpg", "png", "pdf"], key="update_doc_file")
                            if st.button("Save Document", key="update_doc_button"):
                                if update_doc_file:
                                    with st.spinner('Updating document...'):
                                        file_path, text_file_path = update_document(student[2], update_doc_file, matric_number)
                                        st.success(f"Document updated for {student[1]} (Matric: {matric_number}).")
                                        log_activity(st.session_state.user_role, f"Updated document for {matric_number}")
                                else:
                                    st.error("Please upload a document to update.")
                            if not selected_doc.endswith('.txt'):
                                st.download_button(
                                    label="Download Original Document",
                                    data=open(doc_path, "rb").read(),
                                    file_name=selected_doc,
                                    mime="application/octet-stream"
                                )
                            if selected_doc.endswith('.txt'):
                                st.download_button(
                                    label="Download Text Document",
                                    data=open(doc_path, "rb").read(),
                                    file_name=selected_doc,
                                    mime="text/plain"
                                )
                    else:
                        st.info("No documents found for this student.")
                    
                    new_matric_number = st.text_input("Matric Number:", value=student[0], key="update_matric_number")
                    new_name = st.text_input("Name:", value=student[1], key="update_name")
                    new_email = st.text_input("Email:", value=student[5], key="update_email")
                    
                    if st.button("Update Details", key="update_details_button"):
                        update_student_info(new_matric_number, new_name, student[2], student[3], student[4], new_email)
                        st.success("Student details updated successfully.")
                        log_activity(st.session_state.user_role, f"Updated details for {new_matric_number}")
                        if new_email:
                            send_email_notification(new_email, "Student Information Updated", f"Your information has been updated for {new_name} (Matric: {new_matric_number}).")
                    
                    if st.button("Recapture Face Image", key=f"recapture_face_button_{matric_number}"):
                        capture_face_streamlit(student[2])
                        log_activity(st.session_state.user_role, f"Recaptured face image for {new_matric_number}")

                    doc_file = st.file_uploader("Upload Document (Image or PDF)", type=["jpg", "png", "pdf"], key="manage_doc_file")
                    if st.button("Save Document", key="manage_save_doc_button"):
                        if doc_file:
                            with st.spinner('Saving document...'):
                                folder = get_student_folder(matric_number, student[1])
                                file_path, text_file_path = save_document(folder, doc_file, matric_number)
                                st.success(f"Document saved for {student[1]} (Matric: {matric_number}).")
                                log_activity(st.session_state.user_role, f"Uploaded document for {matric_number}")
                       
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
            if st.button("Start Face Capture", key="start_face_capture_button"):
                if st.session_state.verified_student:
                    student_folder = get_student_folder(st.session_state.verified_student[0], st.session_state.verified_student[1])
                    capture_face_streamlit(student_folder)
                else:
                    st.error("Please verify your matric number first.")

        if st.session_state.verified_student:
            matric, name = st.session_state.verified_student
            student = get_student_info(matric)
            
            if student:
                folder = student[2]
                if not verify_face(folder):
                    st.error("Facial recognition failed. Please try again.")
                    return
                st.success(f"Welcome, {student[1]}!")
                documents = list_student_documents(folder)
                if documents:
                    st.subheader("Your Documents")
                    selected_doc = st.selectbox("Select a document to view:", documents)
                    if selected_doc:
                        doc_path = os.path.join(folder, selected_doc)
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
        ai_prompt_page()

def submit_login():
    st.session_state.submit_login = True

if __name__ == "__main__":
    main()
