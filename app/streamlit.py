import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- CONFIGURATION ---
st.set_page_config(
    page_title="NLP Topic Discovery",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Minimalist & Clean) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Cards */
    .metric-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    .badge-gray { background: #f1f5f9; color: #475569; }
    .badge-blue { background: #dbeafe; color: #1e40af; }
    .badge-green { background: #dcfce7; color: #166534; }
    
</style>
""", unsafe_allow_html=True)

# --- SETUP ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    return set(stopwords.words('english')), WordNetLemmatizer()

STOP_WORDS, lemmatizer = setup_nltk()

# --- MODEL LOADING ---
@st.cache_resource
def load_components():
    try:
        # Load from ../models/ as per directory structure
        model = joblib.load('../models/best_model_nmf.joblib')
        vectorizer = joblib.load('../models/tfidf_vectorizer.joblib')
        return model, vectorizer
    except Exception as e:
        return None, None

# --- PIPELINE FUNCTIONS ---
def preprocess_step_by_step(text):
    """Detailed pipeline processing for visualization."""
    steps = {}
    
    # 1. Raw
    steps['raw'] = text
    
    # 2. Cleaning
    clean = text.lower()
    clean = re.sub(r'http\S+|www\S+|urlLink', '', clean)
    clean = re.sub(r'[^a-z\s]', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    steps['cleaned'] = clean
    
    # 3. Tokenization & Stats
    tokens = clean.split()
    token_data = []
    
    lemmatized_tokens = []
    
    for word in tokens:
        lemma = lemmatizer.lemmatize(word)
        is_stop = word in STOP_WORDS
        
        token_data.append({
            "Token": word,
            "Lemma": lemma, 
            "Is Stopword": is_stop,
            "Length": len(word)
        })
        
        if len(lemma) > 2 and not is_stop:
            lemmatized_tokens.append(lemma)
            
    steps['token_df'] = pd.DataFrame(token_data)
    steps['final'] = " ".join(lemmatized_tokens)
    
    return steps

def get_topic_details():
    """Returns fixed mapping for NMF K=5"""
    return {
        0: {"name": "Daily Life & Routine", "color": "#3b82f6"},  # Blue
        1: {"name": "Politics & Government", "color": "#ef4444"},  # Red
        2: {"name": "Social & Chit-Chat", "color": "#eab308"},    # Yellow
        3: {"name": "Personal & Emotions", "color": "#a855f7"},   # Purple
        4: {"name": "Time & Scheduling", "color": "#10b981"}      # Green
    }

# --- UI COMPONENTS ---

def sidebar_info():
    with st.sidebar:
        st.markdown("### 🎓 **UAS NLP Project**")
        st.caption("Universitas - 2025")
        
        st.markdown("---")
        st.markdown("#### **Student Identity**")
        st.markdown("**Name:** Josia Given Santoso")
        st.markdown("**NIM:** 36230035")
        st.markdown("**Class:** 5PDS3")
        
        st.markdown("---")
        st.markdown("#### **Model Stats**")
        st.code("""ModelType: NMF
Topics: 5
Coherence: 0.4597
Engine: Scikit-Learn""", language="yaml")
        
        st.markdown("---")
        st.info("ℹ️ **About:** This dashboard visualizes how unsupervised learning identifies latent topics in blog posts.")

def render_radar_chart(topic_dist, topic_map):
    """Creates a beautiful radar chart for topic distribution."""
    categories = [topic_map[i]['name'] for i in range(len(topic_dist))]
    values = list(topic_dist)
    
    # Close the loop
    categories += [categories[0]]
    values += [values[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line_color='#3b82f6',
        name='Confidence'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values)*1.1]),
        ),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

# --- MAIN LOGIC ---

def main():
    model, vectorizer = load_components()
    topic_map = get_topic_details()

    # Sidebar
    sidebar_info()

    # Main Area
    st.markdown("## 🧠 **Topic Discovery Engine**")
    st.markdown("Analyze the semantic structure of your text using **Non-negative Matrix Factorization**.")
    
    if not model:
        st.error("❌ **Critical Error:** Models not found in `../models/`. Please check directory structure.")
        st.stop()
        
    # Input Area
    with st.container():
        user_input = st.text_area(
            "📝 Enter Text to Analyze",
            height=150,
            placeholder="Type your paragraph here (e.g., 'The president announced a new policy regarding the war...')"
        )
        submit = st.button("🚀 Analyze Text") # Removed use_container_width to be safe

    if submit and user_input:
        start_ts = time.time()
        
        # 1. Detailed Preprocessing
        pipeline_data = preprocess_step_by_step(user_input)
        
        # 2. Inference
        vec = vectorizer.transform([pipeline_data['final']])
        dist = model.transform(vec)[0]
        
        # 3. Stats
        top_idx = np.argmax(dist)
        confidence = np.max(dist)
        norm_confidence = confidence / (np.sum(dist) + 1e-9) # Normalized %
        
        inference_time = time.time() - start_ts
        
        # --- OUTPUT SECTION ---
        st.markdown("---")
        
        # A. Headline Result
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            st.markdown("### 🎯 **Primary Topic Detected**")
            
            # Custom HTML for big result
            st.markdown(f"""
            <div style="background: {topic_map[top_idx]['color']}15; border-left: 6px solid {topic_map[top_idx]['color']}; padding: 20px; border-radius: 8px;">
                <h2 style="color: {topic_map[top_idx]['color']}; margin:0;">{topic_map[top_idx]['name']}</h2>
                <p style="margin-top:8px; color: #475569;">Confidence: <b>{norm_confidence:.1%}</b> • Processing Time: <b>{inference_time:.4f}s</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            st.markdown("#### **🧐 Why this topic?**")
            
            # Feature Importance (Reasoning)
            feat_names = vectorizer.get_feature_names_out()
            topic_vec = model.components_[top_idx]
            
            # Get top words for this topic
            top_words_idx = topic_vec.argsort()[:-15:-1]
            top_topic_words = feat_names[top_words_idx]
            
            # Highlighting logic
            input_tokens = set(pipeline_data['final'].split())
            
            badges_html = ""
            found_relevance = False
            
            for word in top_topic_words:
                if word in input_tokens:
                    badges_html += f'<span class="badge badge-green">✓ {word}</span>'
                    found_relevance = True
                else:
                    badges_html += f'<span class="badge badge-gray">{word}</span>'
            
            st.markdown(badges_html, unsafe_allow_html=True)
            if not found_relevance:
                st.caption("*Note: No direct keyword match found in top 15 features, detection is based on lower-rank semantic overlaps.*")

        with res_col2:
            st.markdown("#### **📊 Topic Probability**")
            fig = render_radar_chart(dist, topic_map)
            st.plotly_chart(fig, use_container_width=True)

        # B. Detailed Pipeline Viewer
        st.markdown("")
        with st.expander("🔍 **View Processing Pipeline (Under the Hood)**", expanded=False):
            
            tab1, tab2, tab3 = st.tabs(["1. Cleaning & Tokens", "2. Vectorization", "3. Raw Data"])
            
            with tab1:
                st.markdown("**Token Analysis:**")
                # Showing dataframe with 'width' parameter instead of use_container_width if warning persists,
                # but standard st.dataframe supports use_container_width in new versions. 
                # User warning specifically mentioned width='stretch'. 
                st.dataframe(pipeline_data['token_df'], use_container_width=True)
                
                st.caption(f"Original Length: {len(pipeline_data['raw'])} chars | Final Tokens: {len(pipeline_data['final'].split())}")

            with tab2:
                st.markdown("**TF-IDF Representation (Non-Zero):**")
                # Show non-zero vectors
                coo = vec.tocoo()
                df_vec = pd.DataFrame({
                    'Term Index': coo.col,
                    'Term': [feat_names[c] for c in coo.col],
                    'TF-IDF Weight': coo.data
                }).sort_values('TF-IDF Weight', ascending=False)
                
                st.dataframe(df_vec, use_container_width=True)
                
            with tab3:
                st.text_area("Preprocessed Text Output:", pipeline_data['final'], disabled=True)

if __name__ == "__main__":
    main()
