import streamlit as st
import os
import cv2
import face_recognition
import sqlite3
import numpy as np
import base64  # For encoding/decoding facial data
from PIL import Image
import imghdr


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
        except Exception as e:
            st.error(f"Error decoding face encoding: {str(e)}")
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

    
def process_uploaded_file(uploaded_file):
    """Handle file upload (saves original only)"""
    try:
        return {
            'original_path': save_student_file(name, matric_number, session, uploaded_file),
            'txt_path': None,
            'docx_path': None
        }
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return None
    
# Ensure a directory for storing student data files
os.makedirs("students_data", exist_ok=True)

# ================= STUDENT FILE SYSTEM MANAGEMENT =================
def save_student_file(name, matric_number, session, program_type, class_level, semester, uploaded_file):
    """Save new student file in organized folder structure"""
    try:
        # Validate inputs
        if not all([name, matric_number, session, semester, uploaded_file]):
            raise ValueError("All fields must be filled")
        
        # Create safe folder names
        def sanitize(text):
            return "".join(c if c.isalnum() or c in (' ', '-', '_') else "_" for c in str(text))
        
        safe_session = sanitize(session)
        safe_program = sanitize(program_type)
        safe_class = sanitize(class_level)
        safe_semester = sanitize(semester)
        safe_name = sanitize(name).replace(" ", "_")
        last_3_matric = str(matric_number)[-3:].zfill(3)  # Ensure 3 digits
        
        # Create folder structure
        base_dir = os.path.abspath("students_data")
        student_dir = os.path.join(base_dir, safe_session, safe_program, safe_class, safe_semester, f"{safe_name}_{last_3_matric}")
        
        os.makedirs(student_dir, exist_ok=False)  # Create student folder if it doesn't exist
        file_path = os.path.join(student_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        raise Exception(f"Error saving file: {str(e)}")

def search_students(search_term, search_by="name", program_type=None, class_level=None, semester=None):
    """Search students with file validation"""
    results = []
    base_dir = os.path.abspath("students_data")
    
    if not os.path.exists(base_dir):
        print("Base directory does not exist.")
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
                if class_level and class_lvl.lower() != class_level.lower():
                    continue
                
                class_dir = os.path.join(program_dir, class_lvl)
                if not os.path.isdir(class_dir):
                    continue
                
                for sem in os.listdir(class_dir):
                    if semester and sem.lower() != semester.lower():
                        continue

                    semester_dir = os.path.join(class_dir, sem)
                    if not os.path.isdir(semester_dir):
                        continue
            
                    for student_folder in os.listdir(semester_dir):
                        student_dir = os.path.join(semester_dir, student_folder)
                        if not os.path.isdir(student_dir):
                            continue
                
                        try:
                            parts = student_folder.rsplit("_", 1)
                            if len(parts) != 2:
                                continue
                            
                            name = parts[0].replace("_", " ")
                            last_3_matric = parts[1]

                            # Debugging: Print extracted details
                            print(f"Checking studnet: {name}, Matric: {last_3_matric}, Session: {session}")
                        
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
                                    "semester": semester,
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
    except Exception as e:
        print(f"Error validating image {filepath}: {str(e)}")
        return False

# Initialize databases
init_db()

if 'authenticated' not in st.session_state:
    st.session_state.clear()
    st.session_state.authenticated = False
    st.query_params["page"] = "Sign In"

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
        if st.sidebar.radio("Are you sure you want to sign out?", ("Yes", "No")) == "Yes":
            st.session_state.clear()
            st.session_state.authenticated = False
            st.query_params.clear()
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
                semester = st.selectbox("Semester", ["First Semester", "Second Semester"], key="semester_select")
                
            with col2:        
                program_type = st.selectbox("Program Type", ["Fulltime", "Parttime"], key="program_type_select")
                class_level = st.selectbox("Class Level", ["ND1", "ND2", "ND3", "HND1", "HND2", "HND3"], key="class_level_select")
                uploaded_file = st.file_uploader("Upload File", type=["pdf", "jpg", "jpeg", "png"], key="file_uploader")
        
            if uploaded_file:
                with st.expander("File Preview"):
                    if uploaded_file.type.startswith("image/"):
                        st.image(uploaded_file, width=300)
                    else:
                        st.write(f"Uploaded file: {uploaded_file.name}")
        
            # Submit button for file upload
            submitted = st.form_submit_button("Upload Student Data")
                       
            if submitted:
                if not all([name, matric_number, session, program_type, class_level, semester, uploaded_file]):
                    st.error("Please fill all required fields")
                else:
                    try:
                        file_path = save_student_file(
                            name, matric_number, session, 
                            program_type, class_level, semester, uploaded_file
                        )
                        st.session_state.upload_success = True
                        st.session_state.file_path = file_path
                        st.success(f"File uploaded successfully: {file_path}")
                                    
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")
                        if 'file_path' in locals() and os.path.exists(file_path):
                            os.remove(file_path)


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
            # Add page navigation
            items_per_page = 10
            total_pages = (len(st.session_state["search_results"]) + items_per_page - 1) // items_per_page
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(st.session_state["search_results"]))

            for idx in range(start_idx, end_idx):
                student = st.session_state["search_results"][idx]
                try:
                    # ADDITIONAL CHECK: Ensure student has files before displaying
                    required_fields = ['matric', 'session', 'program', 'class_level', 'semester']
                    if not all(field in student for field in required_fields):
                        st.warning(f"Skipping incomplete student record: {student.get('name', 'unknown')}")
                        continue

                    #Create a unique session key using all student identifiers
                    session_key = f"show_{student['matric']}_{student['session']}_{student['program']}_{student['class_level']}_{student['semester']}"
                    if session_key not in st.session_state:
                        st.session_state[session_key] = False

                    # Student header with toggle button
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.subheader(f"{student['name']} - {student['matric']}")
                        st.caption(f"Session: {student['session']}")
                        st.caption(f"Program: {student['program']}")
                        st.caption(f"Class Level: {student['class_level']}")
                        st.caption(f"Semester: {student['semester']}")
        
                    with col2:
                        # Create a unique button key by including the index to avoid conflicts
                        button_key = f"toggle_{idx}_{student['matric']}_{student['session']}_{student['program']}_{student['class_level']}_{student['semester']}"
                        # Toggle button to show/hide file details
                        if st.button(
                            "View Details" if not st.session_state.get(session_key, False) else "Hide Details",
                            key=button_key
                        ):
                            st.session_state[session_key] = not st.session_state[session_key]
                            st.rerun()

                    if st.session_state[session_key]:
                        st.markdown("---") # Separator
                        st.write("Files:")
            
                        # Display each file with download option
                        for file_idx, file in enumerate(student['files']):
                            # Display file name and type
                            file_name = os.path.basename(file['path'])
                            file_type = file['type'].upper().replace(".", "")
                            st.write(f"**{file_name}** ({file_type})")
                
                            # Display content based on file type
                            if file['type'] in [".jpg", ".jpeg", ".png"]:
                                st.image(file['path'], width=1000)
                            elif file['type'] == ".pdf":
                                st.warning("PDF content cannot be displayed directly. Download to view.")
                
                            # Download button
                            with open(file['path'], "rb") as f:
                                st.download_button(
                                    label="Download File",
                                    data=f.read(),
                                    file_name=file_name,
                                    key=f"dl_{idx}_{file_idx}_{file['path']}"
                                )
                
                            st.markdown("---") # File separator
                except Exception as e:
                    st.error(f"Error displaying student {student.get('name', 'unknown')}: {str(e)}")
                    continue
                                        

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
