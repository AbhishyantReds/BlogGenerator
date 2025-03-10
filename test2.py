import streamlit as st
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM

## Function to get response from LLaMA 2 model (with GPU)
def getLLamaResponse(input_text, no_words, blog_style):
    try:
        ### Load LLaMA 2 model with GPU
        llm = AutoModelForCausalLM.from_pretrained(
            "models/llama-2-7b-chat.ggmlv3.q8_0.bin",  # Ensure correct path
            model_type="llama",
            gpu_layers=100,  # Use GPU acceleration
            temperature=0.01,
            max_new_tokens=256
        )

        ## Prompt Template
        template = f"Write a blog for {blog_style} job profile on {input_text} within {no_words} words."

        ## Generate response
        response = llm(template)
        return response

    except Exception as e:
        return f"Error: {str(e)}"


# Streamlit UI Setup
st.set_page_config(
    page_title="Generate Blogs",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header("Generate Blogs 🤖")

# User input fields
input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("No of Words", "100")  # Default = 100

with col2:
    blog_style = st.selectbox(
        "Writing the blog for", ("Researchers", "Data Scientist", "Common People"), index=0
    )

submit = st.button("Generate")

## Final response
if submit:
    if input_text.strip():
        try:
            no_words = int(no_words) if no_words.isdigit() else 100  # Convert safely
            response = getLLamaResponse(input_text, no_words, blog_style)
            st.write(response)
        except ValueError:
            st.error("Please enter a valid number for 'No of Words'.")
    else:
        st.error("Please enter a valid blog topic.")
