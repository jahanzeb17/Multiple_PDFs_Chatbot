import os
from dotenv import load_dotenv
import  streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from htmlTemplate import css, bot_template, user_template


load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_kEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = spliter.split_text(text)

    return chunks


def get_vector_store(text_chunks):

    embedding = OpenAIEmbeddings(disallowed_special=())
    vectordb = FAISS.from_texts(texts=text_chunks,embedding=embedding)

    return vectordb



def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model='gpt-3.5-turbo')

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        verbose=True,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversational_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():

    st.set_page_config(page_title="Multiple PDFs RAG app",page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Multiple PDFs RAG app :books:")
    user_query = st.text_input("Ask a question about your documents")
    if user_query:
        handle_userinput(user_query)



    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner('processing'):

                # get text from pdf
                raw_text = get_pdf_text(pdf_docs)

                # get the chunks
                chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vector_store(chunks)
                st.success('Vector store is Ready')

                # create conversational chain
                st.session_state.conversation = get_conversation_chain(vectorstore)




if __name__=="__main__":
    main()