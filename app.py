import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
import re
import json

# =====================================================
# 1. SETUP & STATE
# =====================================================
st.set_page_config(page_title="VedaCure AI", page_icon="🌿", layout="wide")

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False
    if "report_text" not in st.session_state:
        st.session_state.report_text = ""
    if "current_view" not in st.session_state:
        st.session_state.current_view = "chat"
    if "final_diagnosis" not in st.session_state:
        st.session_state.final_diagnosis = None

init_state()

# =====================================================
# 2. STYLING (Professional & Clean)
# =====================================================
st.markdown("""
<style>
    .stApp { background-color: #fcfbf7; }
    
    /* Header */
    .header-box {
        background: linear-gradient(135deg, #1f4e3d 0%, #15803d 100%);
        padding: 30px;
        border-radius: 0 0 30px 30px;
        text-align: center; color: white; margin-bottom: 40px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Chat Bubbles */
    .user-msg {
        background-color: #e6fffa; padding: 15px; border-radius: 20px 20px 5px 20px; 
        margin: 8px 0; border: 1px solid #b2f5ea; float: right; width: 70%; clear: both; color: #0d5f4d;
    }
    .bot-msg {
        background-color: #ffffff; padding: 15px; border-radius: 20px 20px 20px 5px; 
        margin: 8px 0; border: 1px solid #e2e8f0; float: left; width: 75%; clear: both; color: #333;
    }

    /* REPORT: Analysis Box (Fixed height alignment) */
    .reasoning-box {
        background-color: #fff8e1; 
        border-left: 5px solid #ffc107; 
        color: #5d4037; 
        padding: 20px; 
        border-radius: 5px; 
        font-size: 1em;
        height: 100%; 
        display: flex;
        flex-direction: column;
        justify-content: center; /* Ensures text is centered vertically */
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* REPORT: Layout alignment for image + text */
    .report-row {
        display: flex;
        align-items: center;
        gap: 40px;
    }
    .report-col {
        flex: 1;
    }

    /* REPORT: Product Card Grid */
    .product-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        height: 100%; 
        border-top: 5px solid #15803d;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    .product-title {
        color: #15803d; font-weight: bold; font-size: 1.15em; margin-bottom: 8px;
    }
    .product-sub {
        font-size: 0.9em; color: #666; margin-bottom: 10px; font-style: italic;
    }

    /* Buttons */
    .google-btn {
        display: block; width: 100%; text-align: center;
        background-color: #f1f8e9; color: #15803d !important; border: 1px solid #15803d;
        padding: 8px; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 0.85em;
        transition: all 0.2s;
    }
    .google-btn:hover {
        background-color: #15803d; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. BACKEND LOGIC
# =====================================================
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    pass
my_api_key = os.environ.get("GOOGLE_API_KEY")

from huggingface_hub import hf_hub_download
import shutil

HF_REPO_ID = "harshit1272/ayurveda-faiss"
HF_FILE_FAISS = "index.faiss"
HF_FILE_META = "index.pkl"

@st.cache_resource
def load_faiss():
    """
    Download FAISS index files from Hugging Face Hub (if not already cached),
    store them locally, and load them as a LangChain FAISS vectorstore.
    """
    try:
        # Embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Local cache folder
        base_dir = os.path.dirname(os.path.abspath(__file__))
        faiss_dir = os.path.join(base_dir, "faiss_cache")
        os.makedirs(faiss_dir, exist_ok=True)

        # Local file paths
        faiss_path = os.path.join(faiss_dir, HF_FILE_FAISS)
        meta_path = os.path.join(faiss_dir, HF_FILE_META)

        # Download FAISS file if not present
        if not os.path.exists(faiss_path):
            fp = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILE_FAISS,
                token=st.secrets["HF_TOKEN"]
            )
            shutil.copy(fp, faiss_path)

        # Download metadata file if not present
        if not os.path.exists(meta_path):
            fp = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILE_META,
                token=st.secrets["HF_TOKEN"]
            )
            shutil.copy(fp, meta_path)

        # Load FAISS store
        return FAISS.load_local(
            faiss_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    except Exception as e:
        st.error(f"FAISS load failed: {e}")
        return None

db = load_faiss()

def process_report_file(file):
    try:
        if file.type == "application/pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "".join([page.get_text() for page in doc])
        else:
            return pytesseract.image_to_string(Image.open(file))
    except:
        return ""

def generate_structured_report(name, age, gender, feet, inches, weight):
    if not my_api_key:
        return None
    
    with st.spinner("Consulting Ayurvedic texts and generating Dr. Prachi's analysis..."):
        history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        user_symptoms = " ".join([m['content'] for m in st.session_state.messages if m['role'] == 'user'])
        
        # CALCULATE BMI
        bmi = "N/A"
        height_cm = (feet * 30.48) + (inches * 2.54)
        if height_cm > 0 and weight > 0:
            height_m = height_cm / 100
            bmi_val = weight / (height_m ** 2)
            bmi = f"{bmi_val:.1f}"

        # DYNAMIC SEARCH
        if db:
            search_query = f"{user_symptoms} ayurvedic treatment formulation medicine"
            docs = db.similarity_search(search_query, k=6) 
            kb = "\n".join([d.page_content for d in docs])
        else:
            kb = "Database not connected. Use General Classical Ayurveda."

        # HOLISTIC PROMPT WITH BMI
        prompt = f"""
        You are Dr. Prachi, an Expert Ayurvedic Vaidya.
        
        PATIENT PROFILE:
        - Name: {name}
        - Age: {age}
        - Gender: {gender}
        - Height: {feet}'{inches}" ({height_cm} cm)
        - Weight: {weight} kg
        - BMI: {bmi}
        
        CHAT HISTORY: {history}
        UPLOADED REPORT: {st.session_state.report_text[:3000] if st.session_state.report_text else "None"}
        
        RELEVANT SHASTRA TEXT (FROM DATABASE): 
        {kb}
        
        TASKS:
        1. **HOLISTIC ANALYSIS**: Consider Age, BMI (Underweight/Overweight implications on Dosha), and Symptoms. Explain the root cause (Samprapti).
        2. **MEDICATIONS**: Prioritize KB. Must include Ingredients & Dosage.
        3. **FORMATTING**: 
           - Diet/Lifestyle must be bullets.
        
        OUTPUT FORMAT (Strict JSON):
        {{
            "status": "SUCCESS" or "INSUFFICIENT_DATA",
            "reason": "If data missing...",
            "vata": 40, "pitta": 30, "kapha": 30,
            "calculation_logic": "Explain logic considering BMI, age, and symptoms...",
            "diagnosis_summary": "Short summary...",
            "deep_analysis": "Detailed holistic explanation...",
            "diet": "- Point 1\\n- Point 2...",
            "lifestyle": "- Point 1\\n- Point 2...",
            "products": [
                {{ "name": "Name", "formulation": "Ingredients...", "benefit": "Why...", "dosage": "How..." }}
            ]
        }}
        """
        
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.1, google_api_key=my_api_key)
        response = llm.invoke(prompt).content
        
        try:
            json_str = re.search(r"\{.*\}", response, re.DOTALL).group(0)
            return json.loads(json_str)
        except:
            return None

# =====================================================
# 4. SIDEBAR
# =====================================================
with st.sidebar:
    if os.path.exists("drprachi.png"):
        st.image("drprachi.png", use_container_width=True)
    else:
        st.warning("⚠️ drprachi.png not found in folder")
        
    st.markdown("""
        <div style='text-align: center; margin-top: 10px;'>
            <h3 style='color: #15803d; margin-bottom: 5px;'>Dr. Prachi</h3>
            <p style='font-size: 0.9em; color: #666; margin: 0;'>B.A.M.S., MD (Rog Nidan)</p>
            <p style='font-size: 0.8em; color: #888; margin: 0;'>Shri Krishna AYUSH University, Kurukshetra</p>
        </div>
        <hr style='margin-top: 15px; margin-bottom: 15px; border-top: 1px solid #eee;'>
        """, unsafe_allow_html=True)

    # LANGUAGE SELECTION (English + Hinglish + Hindi)
    lang_choice = st.selectbox(
        "Chat Language / भाषा",
        ["English", "Hinglish (Hindi in English)", "Hindi (हिंदी)"]
    )

    st.markdown("### 👤 Holistic Profile")
    p_name = st.text_input("Name", value="User")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        p_age = st.number_input("Age", value=25)
    with col_s2:
        p_weight = st.number_input("Weight (kg)", value=60)
        
    p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown("**Height:**")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        p_feet = st.number_input("Feet", min_value=1, max_value=8, value=5)
    with col_h2:
        p_inches = st.number_input("Inches", min_value=0, max_value=11, value=6)
    
    uploaded = st.file_uploader("Upload Medical Reports", type=["pdf", "jpg", "png"])
    if uploaded and not st.session_state.report_text:
        with st.spinner("Reading File..."):
            st.session_state.report_text = process_report_file(uploaded)
            st.success("✓ Data Extracted")

    st.markdown("---")
    
    if st.session_state.current_view == "chat":
        if not st.session_state.chat_started:
            if st.button("🚀 Start Consultation", use_container_width=True):
                st.session_state.chat_started = True
                st.session_state.messages = []
                
                # GREETING LOGIC BASED ON LANGUAGE
                if "Hinglish" in lang_choice:
                    greeting = (
                        f"Namaste {p_name}. Main Dr. Prachi hoon. "
                        f"Aapki health profile banane ke liye, batayein aaj aapko kya pareshani hai?"
                    )
                elif "Hindi" in lang_choice:
                    greeting = (
                        f"नमस्ते {p_name}। मैं डॉ. प्राची हूँ। "
                        f"आपकी सम्पूर्ण स्वास्थ्य प्रोफ़ाइल बनाने के लिए कृपया बताइए कि आज आपको किस प्रकार की समस्या है?"
                    )
                else:
                    greeting = (
                        f"Namaste {p_name}. I am Dr. Prachi. "
                        f"To build your holistic profile, please describe your main health concern."
                    )
                
                if st.session_state.report_text:
                    greeting += " (I have also reviewed your uploaded report)."
                    
                st.session_state.messages.append({"role": "assistant", "content": greeting})
                st.rerun()
        else:
            st.info("When ready:")
            if st.button("🛍️ Generate Holistic Report", type="primary", use_container_width=True):
                data = generate_structured_report(p_name, p_age, p_gender, p_feet, p_inches, p_weight)
                if data:
                    if data.get("status") == "INSUFFICIENT_DATA":
                        st.error(f"⚠️ {data.get('reason', 'Need more details.')}")
                    else:
                        st.session_state.final_diagnosis = data
                        st.session_state.current_view = "report"
                        st.rerun()
                else:
                    st.error("Error generating report.")

    if st.session_state.current_view == "report":
        if st.button("💬 Back to Chat", use_container_width=True):
            st.session_state.current_view = "chat"
            st.rerun()

    if st.button("🗑️ Reset System"):
        st.session_state.clear()
        st.rerun()

# =====================================================
# 5. MAIN INTERFACE
# =====================================================
st.markdown(
    "<div class='header-box'><h1>🌿 VedaCure AI</h1><p>Authentic Ayurvedic Diagnosis & Formulation</p></div>",
    unsafe_allow_html=True,
)

# -----------------
# VIEW: REPORT
# -----------------
if st.session_state.current_view == "report" and st.session_state.final_diagnosis:
    d = st.session_state.final_diagnosis
    
    st.markdown(f"## 📋 Holistic Health Profile: {p_name}")
    
    # 1. DOSHA VISUALS
    st.subheader("1. Vikriti (Dosha Imbalance)")
    col1, col2, col3 = st.columns(3)
    
    v, p, k = d.get('vata', 0), d.get('pitta', 0), d.get('kapha', 0)
    
    with col1:
        st.markdown(f"**VATA**: {v}%")
        st.progress(v/100)
    with col2:
        st.markdown(f"**PITTA**: {p}%")
        st.progress(p/100)
    with col3:
        st.markdown(f"**KAPHA**: {k}%")
        st.progress(k/100)
    
    st.markdown("---")
    
    # 2. VISUAL REFERENCE & ANALYSIS (ALIGNED)
    st.markdown("<div class='report-row'>", unsafe_allow_html=True)
    c_img, c_text = st.columns([1, 2], gap="large") 
    
    with c_img:
        st.markdown("<div class='report-col'>", unsafe_allow_html=True)
        st.markdown("**Visual Reference:**")
        if os.path.exists("doshas.jpg"):
            st.image("doshas.jpg", caption="Tridosha Balance Chart", width=350)
        else:
            st.info("[Dosha Chart Image Missing: Please add 'doshas.jpg']")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_text:
        st.markdown("<div class='report-col'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='reasoning-box'>
            <div>
                <strong style='color: #5d4037; font-size: 1.1em;'>💡 Dr. Prachi's Analysis:</strong><br><br>
                {d.get('calculation_logic', 'Analysis based on age, weight, and symptoms.')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")

    # 3. DIAGNOSIS
    st.subheader("2. Diagnosis (Nidan)")
    st.info(d.get('diagnosis_summary', 'Analysis Complete.'))
    
    with st.expander("🔍 View Detailed Pathogenesis (Samprapti)"):
        st.markdown(d.get('deep_analysis', 'No detailed analysis available.'))

    st.markdown("---")
    
    # 4. PRODUCTS
    st.subheader("3. Prescribed Formulations (Aushadhi)")
    
    products = d.get('products', [])
    if products:
        cols = st.columns(2)
        for i, prod in enumerate(products):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="product-card">
                    <div class="product-title">{prod['name']}</div>
                    <div class="product-sub">🥣 <b>Formulation:</b> {prod.get('formulation', 'Standard preparation')}</div>
                    <div class="product-benefit"><b>Benefit:</b> {prod['benefit']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"💊 Dosage & Usage"):
                    st.write(f"**Instructions:** {prod.get('dosage', 'As directed by physician.')}")
                
                query = prod['name'].replace(" ", "+")
                link = f"https://www.google.com/search?q=buy+{query}+ayurveda+india"
                st.markdown(
                    f'<a href="{link}" target="_blank" class="google-btn">🛒 Check Availability</a>',
                    unsafe_allow_html=True,
                )
                st.write("") 
    else:
        st.warning("No specific formulations required. Focus on Diet.")

    st.markdown("---")

    # 5. LIFESTYLE & DIET
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown("### 🥗 Pathya (Diet)")
        st.markdown(d.get('diet', '- Balanced meals'))
    with c_b:
        st.markdown("### 🧘 Vihara (Lifestyle)")
        st.markdown(d.get('lifestyle', '- Adequate rest'))

# -----------------
# VIEW: CHAT
# -----------------
elif st.session_state.current_view == "chat":
    if not st.session_state.chat_started:
        st.info("👋 Welcome! Please fill details in the sidebar and click **Start Consultation**.")
    else:
        for msg in st.session_state.messages:
            role = "user-msg" if msg["role"] == "user" else "bot-msg"
            st.markdown(f"<div class='{role}'>{msg['content']}</div>", unsafe_allow_html=True)
            
        if st.button("↩️ Undo Last", key="undo"):
            if len(st.session_state.messages) > 1:
                st.session_state.messages.pop()
                st.session_state.messages.pop()
                st.rerun()

        if user_input := st.chat_input("Type here..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("Dr. Prachi is thinking..."):
                hist = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                
                # CHAT PROMPT
                prompt = f"""
                You are Dr. Prachi, an Expert Ayurvedic Vaidya.
                PATIENT: {p_name} ({p_age} yrs, {p_gender}, {p_feet}'{p_inches}", {p_weight} kg)
                LANGUAGE PREFERENCE: {lang_choice}
                REPORT: {st.session_state.report_text[:1000] if st.session_state.report_text else "None"}
                
                STRATEGY:
                1. **LANGUAGE**: 
                   - If preference is "English", reply in 100% formal English.
                   - If "Hinglish (Hindi in English)", reply in Hinglish (Hindi written in English script).
                   - If "Hindi (हिंदी)", reply fully in Hindi (Devanagari script).
                2. **CRITICAL: NO REPETITIVE GREETINGS**. Do NOT say "Namaste" again.
                3. **KNOWLEDGE LOCK**: If the user asks about ANY non-Ayurvedic topic (e.g., world geography, politics), you MUST politely refuse: "I am Dr. Prachi, an Ayurvedic specialist, and my knowledge is limited to classical texts and holistic health."
                4. **MEDS**: Ask about current medicines if hinted.
                5. **SYMPTOMS**: Drill down 1 level deep.
                
                TEXTS: {{context}}
                HISTORY: {hist}
                """
                
                if db:
                    retriever = db.as_retriever(search_kwargs={"k": 2})
                    llm = ChatGoogleGenerativeAI(
                        model="models/gemini-2.5-flash",
                        temperature=0.1,
                        google_api_key=my_api_key,
                    )
                    chain = (
                        RunnableParallel(context=retriever, question=RunnablePassthrough()) 
                        | PromptTemplate(template=prompt, input_variables=["context", "question"]) 
                        | llm
                    )
                    response = chain.invoke(st.session_state.messages[-1]["content"]).content
                else:
                    response = "System connecting..."

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
