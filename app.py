import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from src.preprocessing import TextPreprocessor
from src.clustering import NetflixClusterer

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Netflix Insight AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to hide default Streamlit branding and adjust margins
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp { margin-top: -30px; }
        .css-1d391kg { padding-top: 1rem; } 
    </style>
""", unsafe_allow_html=True)

# --- 2. HEADER ---
st.title("🎬 Netflix Insight AI")
st.markdown("""
<div style='background-color: #141414; padding: 15px; border-radius: 10px; border: 1px solid #333;'>
    <h4 style='color: #E50914; margin:0;'>Uncover Hidden Genres</h4>
    <p style='margin:0; font-size: 14px; color: #999;'>
        This AI analyzes thousands of plot summaries to reveal semantic categories that Netflix doesn't tell you about.
    </p>
</div>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
    st.header("⚙️ Control Panel")
    
    num_clusters = st.slider("Number of Categories (k)", 5, 25, 15)
    
    st.write("---")
    st.info("💡 **Tip:** Adjust 'k' to see how the AI regroups movies based on plot similarity.")

# --- 4. DATA LOADING (Cached) ---
@st.cache_data
def load_data():
    try:
        # Load data
        df = pd.read_csv('data/netflix_dataset.csv')
        df = df.dropna(subset=['description', 'title']) # Ensure no empty rows
        
        # Preprocess
        preprocessor = TextPreprocessor()
        df['clean_description'] = df['description'].apply(preprocessor.clean_text)
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ Data file not found. Please ensure 'data/netflix_dataset.csv' exists.")
    st.stop()

# --- 5. CLUSTERING LOGIC ---
if 'clusterer' not in st.session_state or st.session_state.get('n_clusters') != num_clusters:
    with st.spinner("🤖 AI is reading movie plots..."):
        clusterer = NetflixClusterer(n_clusters=num_clusters)
        df['cluster_id'] = clusterer.create_clusters(df['clean_description'])
        
        # Store in session state to avoid re-running on every click
        st.session_state['clusterer'] = clusterer
        st.session_state['processed_df'] = df
        st.session_state['n_clusters'] = num_clusters

clusterer = st.session_state['clusterer']
df = st.session_state['processed_df']

# --- 6. MAIN UI TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Interactive Map", "📂 Category Explorer", "🔮 AI Predictor"])

# --- TAB 1: INTERACTIVE PLOTLY CHART ---
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Global Content Map")
        # Prepare PCA for visualization
        pca = PCA(n_components=2)
        matrix = clusterer.vectorizer.transform(df['clean_description'])
        components = pca.fit_transform(matrix.toarray())
        
        # Add PCA coords to dataframe for plotting
        df['pca_x'] = components[:, 0]
        df['pca_y'] = components[:, 1]
        
        # Plotly Scatter Plot
        fig = px.scatter(
            df, 
            x='pca_x', 
            y='pca_y', 
            color=df['cluster_id'].astype(str), # Convert to string so it treats as category, not number
            hover_data=['title', 'description'],
            title=f"Semantic Clustering of {len(df)} Titles",
            template="plotly_dark",
            height=600,
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig.update_layout(showlegend=False) # Hide legend to keep it clean (too many clusters)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Quick Stats")
        st.metric("Total Movies", len(df))
        st.metric("Categories Found", num_clusters)
        st.write("Each dot represents a movie. Movies with similar plots appear closer together.")

# --- TAB 2: CATEGORY EXPLORER ---
with tab2:
    st.subheader(f"Analyzing {num_clusters} Unique Categories")
    
    # Grid Layout for cards
    cols = st.columns(3)
    
    for i in range(num_clusters):
        with cols[i % 3]: # Distribute across 3 columns
            keywords = clusterer.get_cluster_keywords(i, top_n=5)
            
            # Custom HTML Card
            st.markdown(f"""
            <div style="background-color: #222; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #E50914;">
                <h3 style="color: white; margin-top: 0;">Category {i}</h3>
                <p style="color: #bbb; font-size: 14px;"><b>Keywords:</b> {', '.join(keywords)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show top 3 movies in this cluster
            subset = df[df['cluster_id'] == i].head(3)
            for _, row in subset.iterrows():
                st.text(f"• {row['title']}")

# --- TAB 3: PREDICTOR ---
with tab3:
    st.markdown("### Test the Model")
    st.write("Paste a movie plot below, and the AI will determine which existing category it belongs to.")
    
    user_input = st.text_area("Movie Plot Summary:", height=150)
    
    if st.button("Analyze Plot"):
        if user_input:
            # Preprocess
            preprocessor = TextPreprocessor()
            clean_input = preprocessor.clean_text(user_input)
            
            # Transform & Predict
            vec = clusterer.vectorizer.transform([clean_input])
            pred_cluster = clusterer.kmeans.predict(vec)[0]
            
            # Get details
            keywords = clusterer.get_cluster_keywords(pred_cluster)
            
            # Result Card
            st.success(f"Matched to Category {pred_cluster}")
            st.markdown(f"""
                <div style='padding: 20px; background-color: #2E1A1A; border-radius: 10px;'>
                    <h2>Category {pred_cluster}</h2>
                    <p><b>Themes:</b> {', '.join(keywords)}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show similar movies
            st.write("Similar Movies in this category:")
            examples = df[df['cluster_id'] == pred_cluster]['title'].head(5).tolist()
            st.write(examples)
        else:
            st.warning("Please enter some text first.")