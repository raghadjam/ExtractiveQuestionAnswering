# qa_interface.py
import streamlit as st
from retriever import TfidfRetriever

st.title("نظام استرجاع الاجابة - ASCII")

# تحميل الريتريفر
retriever = TfidfRetriever("data.json")

# إدخال المستخدم
question = st.text_input("اكتب سؤالك هنا:")

if question:
    top_context = retriever.retrieve(question)[0]
    st.subheader("أقرب نص من الملف:")
    st.write(top_context)
    st.warning("✅ النموذج لا يستخرج الإجابة بعد. سيتم دمج الإجابة بعد تجهيز النموذج (طالب 2)")
