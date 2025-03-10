import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function to get response from LLaMA 2 model
def getLLamaResponse(input_text, no_words, blog_style):
    try:
        ### LLaMA 2 model
        llm = CTransformers(
            model=r"models\llama-2-7b-chat.ggmlv3.q8_0.bin",  # Fixed path
            model_type="llama",
            config={"max_new_tokens": 256, "temperature": 0.01},
        )

        ## Prompt Template
        template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
        """

        prompt = PromptTemplate(
            input_variables=["blog_style", "input_text", "no_words"],
            template=template,
        )

        ## Generate response from the LLaMA 2 model
        response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
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

## Creating two columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("No of Words", "100")  # Default value = 100

with col2:
    blog_style = st.selectbox(
        "Writing the blog for", ("Researchers", "Data Scientist", "Common People"), index=0
    )

submit = st.button("Generate")

## Final response
if submit:
    if input_text.strip():  # Ensure topic is not empty
        try:
            no_words = int(no_words) if no_words.isdigit() else 100  # Convert words to integer safely
            response = getLLamaResponse(input_text, no_words, blog_style)
            st.write(response)
        except ValueError:
            st.error("Please enter a valid number for 'No of Words'.")
    else:
        st.error("Please enter a valid blog topic.")
