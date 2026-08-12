import os
import requests
# pyrefly: ignore [missing-import]
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Multimodal E-Commerce Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern glassmorphism UI & compact photo cards
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
    }
    .product-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .product-card:hover {
        transform: translateY(-4px);
        border-color: #6366f1;
    }
    .badge-category {
        background-color: #3b82f6;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-tag {
        background-color: #10b981;
        color: white;
        padding: 2px 6px;
        border-radius: 8px;
        font-size: 0.70rem;
        margin-right: 4px;
    }
    .score-badge {
        background-color: #6366f1;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        float: right;
    }
</style>
""", unsafe_allow_html=True)

# API Endpoint URL
API_URL = "http://127.0.0.1:8000/search"

# Header Banner
st.title("🛍️ Multimodal E-Commerce Neural Search Engine")
st.caption("Powered by CLIP Neural Embeddings, FAISS Vector Search, and BM25 Hybrid Fusion across 8,700+ Amazon Products")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Search Controls")
    search_type = st.radio("Select Search Mode:", ["Text Query 🔤", "Visual Image Upload 🖼️"])
    top_k = st.slider("Number of Results (Top-K):", min_value=4, max_value=48, value=12, step=4)
    st.markdown("---")
    st.markdown("### 📊 System Specs")
    st.write("• **Dataset**: 16,957 Physical Photos")    
    st.write("• **Backend**: FastAPI + PyTorch CUDA")
    st.write("• **Vector Index**: 512-D FAISS Matrix")
    st.write("• **Hybrid Search**: BM25 + CLIP RRF")

# Main Interface Area
query_text = None
uploaded_file = None

if "Text Query" in search_type:
    query_text = st.text_input("🔍 Search Catalog (e.g. 'phone cover', 'leather jacket', 'blue shoes'):", value="phone cover")
else:
    uploaded_file = st.file_uploader("🖼️ Upload an Image to find visually similar products:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col_prev, _ = st.columns([1, 3])
        with col_prev:
            st.image(uploaded_file, caption="Uploaded Query Image", width=160)

# Perform Search Request
if st.button("🚀 Search Catalog", type="primary", use_container_width=True):
    with st.spinner("Querying 8,700+ items across FAISS vector matrix..."):
        try:
            results_data = None
            
            if "Text Query" in search_type:
                if not query_text:
                    st.warning("Please enter a search query string.")
                else:
                    response = requests.post(API_URL, data={"text_query": query_text, "k": top_k})
                    if response.status_code == 200:
                        results_data = response.json()
                    else:
                        st.error(f"API Error: {response.text}")
            else:
                if uploaded_file is None:
                    st.warning("Please upload an image file first.")
                else:
                    files = {"image_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(API_URL, files=files, data={"k": top_k})
                    if response.status_code == 200:
                        results_data = response.json()
                    else:
                        st.error(f"API Error: {response.text}")

            if results_data and "results" in results_data:
                meta = results_data.get("meta", {})
                results = results_data.get("results", [])

                st.success(f"Found {len(results)} matches in {meta.get('execution_time_ms', 0):.2f} ms")
                st.markdown("---")

                # Display Results in Responsive 4-Column Grid with Thumbnail Photos
                cols_per_row = 4
                rows = [results[i : i + cols_per_row] for i in range(0, len(results), cols_per_row)]

                for row in rows:
                    cols = st.columns(cols_per_row)
                    for idx, item in enumerate(row):
                        with cols[idx]:
                            product_id = item.get("product_id", "")
                            title = item.get("title", "Product")
                            category = item.get("category", "General")
                            tags = item.get("tags", {})
                            sim_score = item.get("similarity_score", 0.0)

                            # Locate Physical Image File on disk
                            base_dir = os.path.dirname(os.path.abspath(__file__))
                            img_filename = f"{product_id}.jpg"
                            local_path = os.path.join(base_dir, "data", "data_images", img_filename)

                            if os.path.exists(local_path):
                                img = Image.open(local_path)
                            else:
                                img = Image.new("RGB", (200, 200), color=(30, 30, 30))

                            # Thumbnail Photo
                            st.image(img, use_container_width=True)

                            # Product Metadata Card
                            st.markdown(f"**Rank #{item.get('rank')}** | Score: `{sim_score:.3f}`")
                            display_title = (title[:65] + "…") if len(title) > 65 else title
                            st.markdown(f"**{display_title}**")
                            st.markdown(f"<span class='badge-category'>{category}</span>", unsafe_allow_html=True)
                            
                            # AI Zero-Shot Tags
                            tag_str = ""
                            if tags.get("color"):
                                tag_str += f"<span class='badge-tag'>{tags.get('color')}</span>"
                            if tags.get("material"):
                                tag_str += f"<span class='badge-tag'>{tags.get('material')}</span>"
                            if tags.get("style"):
                                tag_str += f"<span class='badge-tag'>{tags.get('style')}</span>"

                            if tag_str:
                                st.markdown(tag_str, unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Failed to connect to backend server: {e}. Make sure 'python -m uvicorn src.api:app --reload' is running!")
