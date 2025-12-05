import streamlit as st
import json
import os
import time
from rag import DiReCTRAG 

@st.cache_data
def check_api_key():
    if "GEMINI_API_KEY" not in os.environ:
        st.error(
            "GEMINI_API_KEY Not Found. "
            "Please set your Gemini API key as an environment variable to use the Gemini API."
        )
        return False
    return True

@st.cache_resource
def load_data(path):
    if not os.path.exists(path):
        st.error(f"Error: Dataset file not found at the specified path: `{path}`. Please check the path and file name.")
        return None
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        st.success(f"Successfully loaded {len(data)} records from: `{path}`.")
        return data
    except json.JSONDecodeError as e:
        st.error(f"Error: Could not decode JSON from `{path}`. Check the file format. Details: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during file loading: {e}")
        return None


@st.cache_resource
def setup_rag_system(data):
    if data is None:
        return None
    try:
        rag_system = DiReCTRAG(data) 
        rag_system._create_index() 
        return rag_system
    except Exception as e:
        st.error(f"Error during RAG system setup: {e}")
        return None


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="DiReCT: Gemini RAG for Clinical Notes",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 DiReCT: RAG for Clinical Reasoning with Gemini")
    st.markdown("""
        **R**etrieval-**A**ugmented **G**eneration (RAG) system using **Gemini** and the **MIMIC-IV-Ext Direct** dataset.
    """)
    st.divider()

    if not check_api_key():
        return
    
    st.header("1. Data Configuration")
    
    file_path = "merged_final.json"
    
    rag_system = None
    
    if file_path:
        
        if 'rag_system' not in st.session_state or st.session_state.rag_system_path != file_path:
            
            if 'rag_system' in st.session_state:
                del st.session_state.rag_system
                
            with st.spinner(f"⏳ Loading dataset from `{file_path}`..."):
                dataset = load_data(file_path)

            if dataset:
                st.subheader("2. RAG System Initialization")
                with st.spinner("⏳ Initializing RAG components and creating FAISS index..."):
                    rag_system = setup_rag_system(dataset)
                    
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.rag_system_path = file_path
                    doc_count = len(rag_system.documents) 
                    st.success(f"RAG System Ready (using BM25). Index has {doc_count} documents.") 
                    st.divider()
                
            
        if 'rag_system' in st.session_state:
            rag_system = st.session_state.rag_system
            st.subheader("2. RAG System Ready")
            st.success(f"RAG System Ready for `{st.session_state.rag_system_path}`.")
            st.divider()
            
    if rag_system:
        st.header("3. Ask a Clinical Question")
        
        user_query = st.text_area(
            "Enter your clinical question or scenario:",
            placeholder="e.g., What are the risk factors for Suspected ACS and what ECG findings characterize STEMI-ACS?",
            height=100,
            key="user_query"
        )

        st.sidebar.header("RAG Parameters")
        k_docs = st.sidebar.slider(
            "Number of documents to retrieve (k):",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
        
        if st.button("Generate Diagnostic Summary", type="primary", use_container_width=True) and user_query:
            
            with st.spinner("Retrieving context and generating response"):
                start_time = time.time()
                
                try:
                    final_answer, retrieved_docs = rag_system.generate_response(
                        query=user_query, 
                        k=k_docs
                    )
                except Exception as e:
                    st.error(f"An error occurred during RAG execution: {e}")
                    final_answer = "Error during generation."
                    retrieved_docs = []

                end_time = time.time()
                processing_time = end_time - start_time

            st.subheader("💡 Generated Answer")
            st.info(final_answer)
            st.caption(f"Processing time: {processing_time:.2f} seconds")
            
            st.subheader(f"📑 Top {len(retrieved_docs)} Retrieved Documents (Context)")
            
            with st.expander("Click to view the source documents used for generation"):
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**{i+1}. Document (Similarity Score: {doc['distance']:.4f})**")
                    st.code(doc['document'], language='markdown')
                    st.markdown("---")
                    
        st.session_state.last_query = user_query
    

if __name__ == "__main__":
    main()



    