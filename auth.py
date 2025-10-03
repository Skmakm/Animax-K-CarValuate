import streamlit as st
import json
import os
import hashlib

USERS_FILE = "users.json"

# ----------------- UTILITIES (No changes needed here) -----------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ----------------- AUTHENTICATION (Final Version) -----------------
def run_authentication():
    # --- PAGE STYLING ---
    st.markdown(
        """
        <style>
        /* This centers the login card on the page */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 80vh;
        }

        /* The authentication card */
        .auth-card {
            max-width: 450px;
            width: 100%;
            margin: auto;
            padding: 2rem; /* Reduced padding slightly for a tighter look */
            border-radius: 10px;
            background: white;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #eef2f6;
        }
        
        /* App Title */
        .app-title {
            text-align: center;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #2b6cb0;
        }
        .app-subtitle {
            text-align: center;
            font-size: 16px;
            color: #718096;
            margin-bottom: 1.5rem;
        }
        
        /* The submit button inside the form */
        .stForm [data-testid="stFormSubmitButton"] button {
            width: 100%;
            border-radius: 8px;
            padding: 10px 0;
            font-weight: 600;
            background-color: #38a169 !important;
            color: white !important;
            border: none;
        }
        .stForm [data-testid="stFormSubmitButton"] button:hover {
            background-color: #2f855a !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
            gap: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "login_user" not in st.session_state:
        st.session_state.login_user = None

    if not st.session_state.authenticated:
        # We create columns to constrain the width of the form
        col1, col2, col3 = st.columns([1, 1.5, 1]) # Adjust ratios as needed for your preference

        with col2: # All the content will be in the middle column
            st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
            
            st.markdown("<div class='app-title'>Animax K CarValuate</div>", unsafe_allow_html=True)
            st.markdown("<div class='app-subtitle'>Please login or signup to continue</div>", unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["🔐 Login", "🆕 Signup"])
            users = load_users()

            # --- LOGIN FORM ---
            with tab1:
                with st.form(key="login_form"):
                    login_username = st.text_input("👤 Username", key="login_username_input")
                    login_password = st.text_input("🔑 Password", type="password", key="login_password_input", autocomplete="new-password")
                    
                    st.write("") # Spacer
                    submitted = st.form_submit_button("Login")
                    if submitted:
                        if login_username in users and users[login_username] == hash_password(login_password):
                            st.session_state.authenticated = True
                            st.session_state.login_user = login_username
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")

            # --- SIGNUP FORM ---
            with tab2:
                with st.form(key="signup_form"):
                    new_username = st.text_input("👤 Choose a Username", key="signup_username_input")
                    new_password = st.text_input("🔑 Choose a Password", type="password", key="signup_password_input", autocomplete="new-password")
                    confirm_password = st.text_input("🔑 Confirm Password", type="password", key="signup_confirm_input", autocomplete="new-password")
                    
                    st.write("") # Spacer
                    submitted = st.form_submit_button("Signup")
                    if submitted:
                        if not new_username or not new_password:
                            st.warning("⚠️ Please fill out all fields.")
                        elif new_username in users:
                            st.error("❌ Username already exists.")
                        elif new_password != confirm_password:
                            st.error("❌ Passwords do not match.")
                        elif len(new_password) < 6:
                            st.error("❌ Password must be at least 6 characters long.")
                        else:
                            users[new_username] = hash_password(new_password)
                            save_users(users)
                            st.success("✅ Account created! Please login.")
                            
            st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # --- LOGOUT ---
    st.sidebar.success(f"👤 Logged in as **{st.session_state.login_user}**")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.login_user = None
        st.rerun()