import streamlit as st
from mcq_generator import ImprovedMCQGenerator, is_suitable_for_students, format_mcq

st.set_page_config(page_title="MCQ Generator", page_icon="📚", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>📚 Intelligent MCQ Generator for Educational Content</h1>
""", unsafe_allow_html=True)

st.sidebar.header("Settings")
num_questions = st.sidebar.slider("Number of MCQs to Generate", 1, 10, 5)

theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #2e2e2e; color: white; }
        .stTextInput>div>div>input { background-color: #3e3e3e; color: white; }
        </style>
    """, unsafe_allow_html=True)

st.write("### Upload a .txt file or enter text manually:")
uploaded_file = st.file_uploader("Upload Text File", type=["txt"])

if uploaded_file is not None:
    paragraph = uploaded_file.read().decode("utf-8")
    st.success("✅ Text file loaded!")
else:
    paragraph = st.text_area("Or type/paste your paragraph here:", height=300)

if st.button("🚀 Generate MCQs"):
    if not paragraph.strip():
        st.warning("⚠️ Please provide some text to generate MCQs.")
    else:
        with st.spinner("Checking content suitability..."):
            if is_suitable_for_students(paragraph):
                generator = ImprovedMCQGenerator()
                mcqs = generator.generate_mcqs(paragraph, num_questions)

                if mcqs:
                    st.success(f"✅ Successfully generated {len(mcqs)} MCQs!")
                    for i, mcq in enumerate(mcqs):
                        with st.expander(f"Question {i+1}: {mcq['question']}"):
                            for idx, option in enumerate(mcq['options']):
                                st.markdown(f"- {chr(65+idx)}. {option}")
                            st.markdown(f"**Answer:** {chr(65 + mcq['answer_index'])}")

                    all_mcqs_text = "\n\n".join([format_mcq(mcq, i) for i, mcq in enumerate(mcqs)])
                    st.download_button(
                        label="📥 Download MCQs as Text File",
                        data=all_mcqs_text,
                        file_name="generated_mcqs.txt",
                        mime="text/plain",
                        help="Download all generated MCQs"
                    )
                else:
                    st.error("⚠️ Could not generate MCQs. Try with a longer or more informative paragraph.")
            else:
                st.error("❌ Content is not suitable for educational MCQ generation. Please revise the text.")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit | Powered by T5 and BART models</p>", unsafe_allow_html=True)
