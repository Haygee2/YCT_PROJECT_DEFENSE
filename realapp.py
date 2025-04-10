import streamlit as st
import requests
import json
import cv2
import face_recognition
import sqlite3
from io import StringIO
import docx
from docx.shared import Pt
import pytesseract
import io
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import numpy as np
import base64  # For encoding/decoding facial data
import os
from pathlib import Path
from PIL import Image
import tempfile
import imghdr
import time

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# SQLite database setup
def init_db():
    conn = sqlite3.connect("admins.db")
    cursor = conn.cursor()

    # Check if the face_encoding column exists
    cursor.execute("PRAGMA table_info(admins)")
    columns = [column[1] for column in cursor.fetchall()]
    if "face_encoding" not in columns:
        # Backup existing data
        cursor.execute("SELECT username, password FROM admins")
        existing_data = cursor.fetchall()

        # Recreate the table with the new schema
        cursor.execute("DROP TABLE IF EXISTS admins")
        cursor.execute("""
            CREATE TABLE admins (
                username TEXT PRIMARY KEY,
                password TEXT,
                face_encoding TEXT
            )
        """)

        # Restore existing data
        for username, password in existing_data:
            cursor.execute("INSERT INTO admins (username, password) VALUES (?, ?)", (username, password))

    conn.commit()
    conn.close()

def add_admin_to_db(username, password, face_encoding):
    conn = sqlite3.connect("admins.db")
    cursor = conn.cursor()
    
    if isinstance(face_encoding, np.ndarray):
        face_encoding_str = base64.b64encode(face_encoding.tobytes()).decode("utf-8")
    else:
        face_encoding_str = base64.b64encode(face_encoding).decode("utf-8")
    cursor.execute("INSERT INTO admins (username, password, face_encoding) VALUES (?, ?, ?)", 
                   (username, password, face_encoding_str))
    conn.commit()
    conn.close()

def get_admin_from_db(username):
    conn = sqlite3.connect("admins.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password, face_encoding FROM admins WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        password, face_encoding_str = result
        if face_encoding_str is None:  # Handle NULL face_encoding
            return password, None
        try:
            face_encoding = np.frombuffer(base64.b64decode(face_encoding_str), dtype=np.float64)  # Decode face encoding
            return password, face_encoding
        except:
            return password, None
    return None, None

def delete_admin_from_db(username):
    conn = sqlite3.connect("admins.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM admins WHERE username = ?", (username,))
    conn.commit()
    conn.close()

# Facial recognition functions using face_recognition
def capture_face():
    st.write("Initializing webcam for face capture...")
    cap = cv2.VideoCapture(0)
    st.write("Press 'c' to capture your face.")
    face_encoding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        cv2.imshow("Face Capture", frame)
        key = cv2.waitKey(1)

        if key == ord('c'):  # Press 'c' to capture
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 1:  # Ensure only one face is captured
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                st.write("Face captured successfully!")
            else:
                st.error("Please ensure only one face is visible.")
            break
        elif key == ord('q'):  # Press 'q' to quit
            st.write("Face capture canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return face_encoding

def verify_face(stored_encoding):
    st.write("Initializing webcam for face verification...")
    cap = cv2.VideoCapture(0)
    st.write("Press 'v' to verify your face.")
    verified = False

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        cv2.imshow("Face Verification", frame)
        key = cv2.waitKey(1)

        if key == ord('v'):  # Press 'v' to verify
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 1:  # Ensure only one face is visible
                current_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                # Use a tolerance value for robust comparison
                verified = face_recognition.compare_faces([stored_encoding], current_encoding, tolerance=0.6)[0]
                if verified:
                    st.write("Face verification successful!")
                else:
                    st.error("Face verification failed.")
            else:
                st.error("Please ensure only one face is visible.")
            break
        elif key == ord('q'):  # Press 'q' to quit
            st.write("Face verification canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return verified  # Make sure this return statement is at the end of the function

def is_valid_image(image_bytes):
    """Verify if the uploaded file is a valid image"""
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
        return True
    except:
        return False

def extract_text_from_image(image_bytes):
    """Extract text from image using OCR"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed (Tesseract expects RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        st.error(f"Image extraction failed: {str(e)}")
        return None
    
def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF using OCR"""
    try:
        images = convert_from_bytes(pdf_bytes)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF extraction failed: {str(e)}")
        return None
    
def clean_extracted_text_with_ai(raw_text):
    """Use AI to clean and structure extracted text with robust error handling"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Truncate very long text to stay within token limits
        max_length = 12000  # OpenRouter's limit for most models
        if len(raw_text) > max_length:
            raw_text = raw_text[:max_length]
            st.warning("Document was truncated to fit AI processing limits")
        
        prompt = f"""
        Clean and structure this academic document exactly as it appears in the original.
        PRESERVE ALL FORMATTING, SPACING, and ALIGNMENT.
        Only correct obvious OCR errors when absolutely certain.
        Maintain all numbers, grades, and codes exactly as shown.
        
        Input Document:
        {raw_text}
        
        Cleaned Output:
        """
        
        payload = {
            "model": "deepseek/deepseek-r1:free" or "deepseek/deepseek-v3-base:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,  # Lower temperature for more consistent results
            "max_tokens": 4000
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30  # Added timeout
        )
        
        # Check for successful response
        if response.status_code == 200:
            response_data = response.json()
            
            # Handle different response formats
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            elif "message" in response_data:  # Some APIs use this format
                return response_data["message"]["content"]
            else:
                st.error("Unexpected API response format")
                return raw_text  # Fallback to original text
        
        # Handle API errors
        else:
            error_msg = response.text
            try:
                error_detail = response.json().get("error", {}).get("message", error_msg)
            except:
                error_detail = error_msg
            
            st.error(f"AI processing failed (Status {response.status_code}): {error_detail}")
            return raw_text  # Fallback to original text
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error during AI processing: {str(e)}")
        return raw_text
    except Exception as e:
        st.error(f"Unexpected error during AI cleaning: {str(e)}")
        return raw_text
    
def process_uploaded_file(uploaded_file, extract_txt=True, extract_docx=True):
    """Handle both PDF and image files"""
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type
    
    results = {'original_path': None, 'txt_path': None, 'docx_path': None}
    
    try:
        # Save original file first
        results['original_path'] = save_student_file(name, matric_number, session, uploaded_file)
        
        # Process based on file type
        if file_type == "application/pdf":
            raw_text = extract_text_from_pdf(file_bytes)
        elif file_type in ["image/jpeg", "image/png", "image/jpg"]:
            if not is_valid_image(file_bytes):
                raise ValueError("Invalid image file")
            raw_text = extract_text_from_image(file_bytes)
        else:
            raise ValueError("Unsupported file type")
        
        if raw_text:
            cleaned_text = clean_extracted_text_with_ai(raw_text)
            
            if extract_txt:
                results['txt_path'] = save_as_text_file(cleaned_text, results['original_path'])
            
            if extract_docx:
                results['docx_path'] = save_as_word_file(cleaned_text, results['original_path'])
        
        return results
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        # Clean up partially created files
        for path in results.values():
            if path and os.path.exists(path):
                os.remove(path)
        return None
    
def save_as_text_file(text_content, original_file_path):
    """Save extracted text as .txt file in same directory"""
    try:
        base_path = os.path.splitext(original_file_path)[0]
        txt_path = f"{base_path}_extracted.txt"
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        return txt_path
    except Exception as e:
        st.error(f"Error saving text file: {str(e)}")
        return None
    
def save_as_word_file(text_content, original_file_path):
    """Save extracted text as .docx file"""
    try:
        base_path = os.path.splitext(original_file_path)[0]
        docx_path = f"{base_path}_extracted.docx"
        
        doc = docx.Document()
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Courier New'  # Monospace for alignment preservation
        font.size = Pt(10)
        
        for line in text_content.split('\n'):
            doc.add_paragraph(line)
        
        doc.save(docx_path)
        return docx_path
    except Exception as e:
        st.error(f"Error saving Word file: {str(e)}")
        return None  

# Ensure a directory for storing student data files
os.makedirs("students_data", exist_ok=True)

# ================= STUDENT FILE SYSTEM MANAGEMENT =================
def save_student_file(name, matric_number, session, program_type, class_level, uploaded_file):
    """Save new student file in organized folder structure"""
    try:
        # Validate inputs
        if not all([name, matric_number, session, uploaded_file]):
            raise ValueError("All fields must be filled")
        
        # Create safe folder names
        def sanitize(text):
            return "".join(c if c.isalnum() or c in (' ', '-', '_') else "_" for c in str(text))
        
        safe_session = sanitize(session)
        safe_program = sanitize(program_type)
        safe_class = sanitize(class_level)
        safe_name = sanitize(name).replace(" ", "_")
        last_3_matric = str(matric_number)[-3:].zfill(3)  # Ensure 3 digits
        
        # Create folder structure
        base_dir = os.path.abspath("students_data")
        student_dir = os.path.join(base_dir, safe_session, safe_program, safe_class, f"{safe_name}_{last_3_matric}")
        
        os.makedirs(student_dir, exist_ok=False)  # Create student folder if it doesn't exist
        file_path = os.path.join(student_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
         # Extract and save text versions for supported file types
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext in ['.pdf', '.jpg', '.jpeg', '.png']:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            
            if file_ext == '.pdf':
                raw_text = extract_text_from_pdf(file_bytes)
            else:
                raw_text = extract_text_from_image(file_bytes)
            
            if raw_text:
                cleaned_text = clean_extracted_text_with_ai(raw_text)
                if cleaned_text:
                    save_as_text_file(cleaned_text, file_path)
                    save_as_word_file(cleaned_text, file_path)
            
        return file_path
    except Exception as e:
        raise Exception(f"Error saving file: {str(e)}")

def search_students(search_term, search_by="name", program_type=None, class_level=None):
    """Search students with file validation"""
    results = []
    base_dir = os.path.abspath("students_data")
    
    if not os.path.exists(base_dir):
        return results
    
    for session in os.listdir(base_dir):
        session_dir = os.path.join(base_dir, session)
        if not os.path.isdir(session_dir):
            continue
        
        for program in os.listdir(session_dir):
            if program_type and program.lower() != program_type.lower():
                continue
            
            program_dir = os.path.join(session_dir, program)
            if not os.path.isdir(program_dir):
                continue
            
            for class_lvl in os.listdir(program_dir):
                if class_lvl and class_lvl.lower() != class_lvl.lower():
                    continue
                
                class_dir = os.path.join(program_dir, class_lvl)
                if not os.path.isdir(class_dir):
                    continue
            
                for student_folder in os.listdir(class_dir):
                    student_dir = os.path.join(class_dir, student_folder)
                    if not os.path.isdir(student_dir):
                        continue
                
                    try:
                        parts = student_folder.rsplit("_", 1)
                        if len(parts) != 2:
                            continue
                            
                        name = parts[0].replace("_", " ")
                        last_3_matric = parts[1]
                        
                        # Search matching
                        match = False
                        search_term = str(search_term).lower()
                        
                        if search_by == "name" and search_term in name.lower():
                            match = True
                        elif search_by == "matric" and search_term in last_3_matric:
                            match = True
                        elif search_by == "session" and search_term in session.lower():
                            match = True
                            
                        if match:
                            valid_files = []
                            for filename in os.listdir(student_dir):
                                file_path = os.path.join(student_dir, filename)
                                if os.path.isfile(file_path):
                                    file_type = os.path.splitext(filename)[1].lower()
                                    
                                    if file_type in [".jpg", ".jpeg", ".png"] and not is_valid_image(file_path):
                                        continue
                                        
                                    valid_files.append({
                                        "path": file_path,
                                        "type": file_type
                                    })
                            
                            results.append({
                                "name": name,
                                "matric": last_3_matric,
                                "session": session,
                                "program": program,
                                "class_level": class_lvl,
                                "files": valid_files,
                                "has_files": bool(valid_files)
                            })
                    except Exception as e:
                        print(f"Error processing {student_folder}: {str(e)}")
                        continue
                        
    return results

def is_valid_image(filepath):
    """Check if file is a valid image"""
    try:
        # First check the file header
        image_type = imghdr.what(filepath)
        if image_type not in ['jpeg', 'png', 'gif', 'bmp']:
            return False
        
        # Then try to open with PIL
        with Image.open(filepath) as img:
            img.verify()
        return True
    except:
        return False

# Initialize databases
init_db()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Check query parameters to determine the current page
query_params = st.query_params
current_page = st.session_state.get("current_page", query_params.get("page", ["Sign In"])[0])

# Check if the user is authenticated
if current_page == "Admin Dashboard" and not st.session_state.authenticated:
    st.warning("Please sign in to access the admin dashboard.")
    st.query_params["page"]= "Sign In"
    st.rerun()

# Authentication check at the start
if current_page == "Admin Dashboard":
    # ===== AUTHENTICATED ADMIN DASHBOARD =====
    st.sidebar.title("Admin Dashboard")
    admin_page = st.sidebar.selectbox("Menu", [
        "Upload Student Data", 
        "Search Student Data",
    ])
    
        # Sign Out button
    if st.sidebar.button("Sign Out"):
        st.session_state.authenticated = False
        st.query_params["page"] = "Sign In"
        st.rerun()

        # Admin Dashboard Pages
    if admin_page == "Upload Student Data":
        st.title("Upload Student Data")
        
        if 'download_docx' not in st.session_state:
            st.session_state.download_docx = False
            
        with st.form("upload_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                # Student details input fields
                name = st.text_input("Student Name", key="student_name")
                matric_number = st.text_input("Matric Number", key="matric_number")
                session = st.text_input("Session (e.g., 2022-2023)", key="session")
                
            with col2:        
                program_type = st.selectbox("Program Type", ["Fulltime", "Parttime"], key="program_type")
        
                if program_type == "Fulltime":
                    class_level = st.selectbox("Class Level", ["ND1", "ND2", "HND1", "HND2"], key="fulltime_class")
                else:
                    class_level = st.selectbox("Class Level", ["ND1", "ND2", "ND3", "HND1", "HND2", "HND3"], key="parttime_class")
        
                uploaded_file = st.file_uploader("Upload File", type=["pdf", "jpg", "jpeg", "png"], key="file_uploader")
        
            if uploaded_file:
                with st.expander("File Preview"):
                    if uploaded_file.type.startswith("image/"):
                        st.image(uploaded_file, width=300)
                    else:
                        st.write(f"PDF file: {uploaded_file.name}")
                        
                col1, col2 = st.columns(2)
                with col1:
                    extract_txt = st.checkbox("Extract as Text", value=True, key="extract_txt")
                with col2:
                    extract_docx = st.checkbox("Extract as Word", value=True, key="extract_docx")
        
            # Submit button for file upload
            submitted = st.form_submit_button("Upload Student Data")
                       
            if submitted:
                if not all([name, matric_number, session, program_type, class_level, uploaded_file]):
                    st.error("Please fill all required fields")
                else:
                    try:
                        file_path = save_student_file(
                            name, matric_number, session, 
                            program_type, class_level, uploaded_file
                        )
                        st.session_state.upload_success = True
                        st.session_state.file_path = file_path
                        st.success(f"File uploaded successfully: {file_path}")
                    
                        if extract_txt or extract_docx:
                            with st.spinner("Extracting text content..."):
                                with open(file_path, "rb") as f:
                                    file_bytes = f.read()
                        
                                file_type = uploaded_file.type
                                raw_text = None
                    
                                try:
                                    if file_type == "application/pdf":
                                        # NEW: PDF-specific extraction with fallback
                                        try:
                                            raw_text = extract_text_from_pdf(file_bytes)
                                            if not raw_text or len(raw_text) < 10:  # If extraction seems incomplete
                                                raise Exception("Direct extraction failed - trying OCR")
                                        except:
                                            st.warning("Using OCR for PDF text extraction...")
                                            raw_text = extract_text_from_image(file_bytes)  # OCR fallback
                                    elif file_type.startswith("image/"):
                                        raw_text = extract_text_from_image(file_bytes)
                                except Exception as e:
                                    st.error(f"Extraction error: {str(e)}")
                
                                if raw_text:
                                    with st.spinner("Cleaning text with AI..."):
                                        cleaned_text = clean_extracted_text_with_ai(raw_text)
                    
                                    if extract_txt:
                                        txt_path = save_as_text_file(cleaned_text, file_path)
                                        st.success(f"Text version saved: {txt_path}")

                    
                                    if extract_docx:
                                        docx_path = save_as_word_file(cleaned_text, file_path)
                                        st.session_state.docx_path = docx_path
                                        st.session_state.download_docx = True
                                        st.success(f"Word version saved: {docx_path}")

                                else:
                                    st.error("Text extraction failed")
                
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")
                        if 'file_path' in locals() and os.path.exists(file_path):
                            os.remove(file_path)
                            
        #Download button for the extracted Word file
        if st.session_state.get('download_docx') and st.session_state.get('docx_path'):
            with open(st.session_state.docx_path, "rb") as f:
                st.download_button(
                    "Download Extracted Word File",
                    f.read(),
                    file_name=os.path.basename(st.session_state.docx_path),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="word_download"
                )

    # Search Student Data section
    elif admin_page == "Search Student Data":
        st.title("Search Student Data")
        with st.form("search_form"):
            search_term = st.text_input("Enter search term")
            search_by = st.selectbox("Search by", ["Name", "Matric (last 3 digits)", "Session"])
    
            # New search options
            program_type = st.selectbox("Filter by Program Type", ["ALL", "Fulltime", "Parttime"])
            class_level = st.selectbox("Filter by Class Level", ["ALL", "ND1", "ND2", "ND3", "HND1", "HND2", "HND3"])
    
            search_submitted = st.form_submit_button("Search")

        if search_submitted:
            if not search_term.strip():
                st.warning("Please enter a search term")
            else:
                search_mapping = {
                    "Name": "name",
                    "Matric (last 3 digits)": "matric",
                    "Session": "session"
                }
        
                # Convert "ALL" to None for filtering
                program_filter = None if program_type == "ALL" else program_type
                class_filter = None if class_level == "ALL" else class_level
        
                results = search_students(search_term, search_mapping[search_by], program_filter, class_filter)
                st.session_state["search_results"] = results  # Store results in session state

        if "search_results" in st.session_state:
            for student in st.session_state["search_results"]:
                # Initialize session state if not exists
                session_key = f"show_{student['matric']}"
                if session_key not in st.session_state:
                    st.session_state[session_key] = False

                # Student header with toggle button
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader(f"{student['name']} - {student['matric']}")
                    st.caption(f"Session: {student['session']}")
            
                with col2:
                    if st.button(
                        "View Details" if not st.session_state[session_key] else "Hide Details",
                        key=f"toggle_{student['matric']}"
                    ):
                        st.session_state[session_key] = not st.session_state[session_key]
                        st.rerun()  # Add this to force update

                if st.session_state[session_key]:
                    st.markdown("---")  # Visual separator
                    if student['has_files']:
                        for file in student['files']:
                            st.subheader(os.path.basename(file['path']))
            
                            # Tabs for different views
                            tab1, tab2, tab3 = st.tabs(["📄 Original", "✍️ Text", "⚙️ Actions"])
            
                            with tab1:
                                if file['type'] == ".pdf":
                                    try:
                                        from PyPDF2 import PdfReader
                                        with open(file['path'], "rb") as f:
                                            pdf = PdfReader(f)
                                            num_pages = len(pdf.pages)
                        
                                        if num_pages > 1:
                                            page_num = st.selectbox(
                                                "Select page",
                                                range(1, num_pages+1),
                                                key=f"page_{file['path']}"
                                            ) - 1
                                        else:
                                            page_num = 0
                        
                                        page = pdf.pages[page_num]
                                        st.text(page.extract_text())
                                    except Exception as e:
                                        st.warning(f"Couldn't preview PDF: {str(e)}")
                                elif file['type'] in [".jpg", ".jpeg", ".png"]:
                                    if is_valid_image(file['path']):
                                        st.image(file['path'], use_column_width=True)
                                    else:
                                        st.warning("Corrupted image file")
            
                            with tab2:
                                base_path = os.path.splitext(file['path'])[0]
                                txt_path = f"{base_path}_extracted.txt"
            
                                if os.path.exists(txt_path):
                                    with open(txt_path, "r", encoding="utf-8") as f:
                                        text_content = f.read()
                                        search_text = st.text_input("Search in text", key=f"search_{file['path']}")
                    
                                        if search_text:
                                            highlighted = text_content.replace(
                                                search_text,
                                                f"**{search_text}**"
                                            )
                                            st.markdown(highlighted, unsafe_allow_html=True)
                                        
                                        else:
                                            st.text_area(
                                                "Extracted Text", 
                                                text_content,
                                                height=300,
                                                key=f"preview_{file['path']}"
                                            )
                                else:
                                    st.warning("No extracted text available")
                                    if st.button("✨ Extract Text Now", key=f"quick_extract_{file['path']}"):
                                        with st.spinner("Extracting..."):
                                            with open(file['path'], "rb") as f:
                                                file_bytes = f.read()
                                                
                                            if file['type'] == '.pdf':
                                                raw_text = extract_text_from_pdf(file_bytes)
                                            else:
                                                raw_text = extract_text_from_image(file_bytes)

                                            if raw_text:
                                                cleaned_text = clean_extracted_text_with_ai(raw_text)
                                                save_as_text_file(cleaned_text, file['path'])
                                                save_as_word_file(cleaned_text, file['path'])
                                                st.rerun()
                                            else:
                                                st.error("Extraction failed")
            
                            with tab3:
                                base_path = os.path.splitext(file['path'])[0]
                                txt_path = f"{base_path}_extracted.txt"
                                docx_path = f"{base_path}_extracted.docx"

                                # Actions column layout
                                col1, col2 = st.columns(2)

                                with col1:
                                    if os.path.exists(txt_path):
                                        with open(txt_path, "r", encoding="utf-8") as f:
                                            st.download_button(
                                                "📥 Download Text",
                                                f.read(),
                                                file_name=f"{os.path.basename(base_path)}.txt",
                                                key=f"txt_{file['path']}"
                                            )

                                with col2:
                                    if os.path.exists(docx_path):
                                        with open(docx_path, "rb") as f:
                                            st.download_button(
                                                "📝 Download Word",
                                                f.read(),
                                                file_name=f"{os.path.basename(base_path)}.docx",
                                                key=f"docx_{file['path']}"
                                            )

                                # Advanced options
                                st.markdown("### Advanced Options")
                                if st.button("🔄 Re-extract Text", key=f"re_extract_{file['path']}"):
                                    with st.spinner("Re-extracting..."):
                                        with open(file['path'], "rb") as f:
                                            file_bytes = f.read()

                                        if file['type'] == '.pdf':
                                            raw_text = extract_text_from_pdf(file_bytes)
                                        else:
                                            raw_text = extract_text_from_image(file_bytes)

                                        if raw_text:
                                            cleaned_text = clean_extracted_text_with_ai(raw_text)
                                            save_as_text_file(cleaned_text, file['path'])
                                            save_as_word_file(cleaned_text, file['path'])
                                            st.success("Re-extraction complete!")
                                            st.rerun()
                                        else:
                                            st.error("Extraction failed")
                    else:
                        st.error("No matching records found")

else:
    # ===== AUTHENTICATION PAGES =====
    st.sidebar.title("Authentication")
    page = st.sidebar.selectbox("Menu", ["Sign In", "Sign Up", "Delete Admin"])

    if page == "Sign In":
        st.title("Sign In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Sign In"):
            stored_password, stored_face_encoding = get_admin_from_db(username)
            if stored_password and stored_password == password:
                if stored_face_encoding is None or verify_face(stored_face_encoding):
                    st.session_state.authenticated = True
                    st.session_state.current_page = "Admin Dashboard"
                    st.query_params["page"] = "Admin Dashboard"
                    st.rerun()
                else:
                    st.error("Facial verification failed.")
            else:
                st.error("Invalid username or password.")

    elif page == "Sign Up":
        st.title("Sign Up")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        
        if st.button("Sign Up"):
            stored_password, _ = get_admin_from_db(new_username)
            if stored_password:
                st.error("Username already exists.")
            else:
                face_encoding = capture_face()
                if face_encoding is not None:
                    add_admin_to_db(new_username, new_password, face_encoding.tobytes())
                    st.success("Sign Up successful! Facial data captured.")
                else:
                    st.error("Face capture failed. Please try again.")

    elif page == "Delete Admin":
        st.title("Delete Admin")
        del_username = st.text_input("Username")
        del_password = st.text_input("Password", type="password")
        
        if st.button("Delete Admin"):
            stored_password, stored_face_encoding = get_admin_from_db(del_username)
            if stored_password and stored_password == del_password:
                if stored_face_encoding is None or verify_face(stored_face_encoding):
                    delete_admin_from_db(del_username)
                    st.success("Admin deleted successfully!")
                else:
                    st.error("Facial verification failed. Please try again.")
            else:
                st.error("Invalid username or password.")