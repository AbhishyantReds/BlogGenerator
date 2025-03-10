import streamlit as st
import time
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM
from io import BytesIO

def getLLamaResponse(input_text, no_words, blog_style):
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "models/llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            gpu_layers=100,
            temperature=0.01,
            max_new_tokens=512  # Increased for better output
        )
        
        template = f"Write a blog for {blog_style} job profile on {input_text} within {no_words} words."
        response = llm(template)
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"

st.set_page_config(
    page_title="AI Blog Generator - Tenor",
    page_icon="📝",
    layout="centered",
)

st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4A90E2;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: #666;
        }
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #357ABD;
        }
        .tenor-team {
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>AI Blog Generator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Powered by LLaMA 2 | Team Tenor 🚀</div>", unsafe_allow_html=True)

input_text = st.text_input("Enter Blog Topic")
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("No of Words", "100")

with col2:
    blog_style = st.selectbox("Writing for", ("Researchers", "Data Scientist", "Common People"), index=0)

submit = st.button("Generate Blog 📝")

if submit:
    if input_text.strip():
        with st.spinner("Generating blog... Please wait ⏳"):
            try:
                no_words = int(no_words) if no_words.isdigit() else 100
                response = getLLamaResponse(input_text, no_words, blog_style)
                
                if "Error:" in response:
                    st.error(response)
                else:
                    st.success("Blog Generated Successfully! ✅")
                    st.write(response)
                    
                    # Convert blog to downloadable text file
                    text_bytes = BytesIO(response.encode("utf-8"))
                    st.download_button(
                        label="Download Blog as Text File 📥",
                        data=text_bytes,
                        file_name=f"{input_text.replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
            
            except ValueError:
                st.error("Please enter a valid number for 'No of Words'.")
    else:
        st.error("Please enter a valid blog topic.")

st.markdown("<div class='tenor-team'>© 2025 Team Tenor - All Rights Reserved</div>", unsafe_allow_html=True)
