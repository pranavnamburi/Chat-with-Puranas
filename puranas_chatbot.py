import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models.openai import ChatOpenAI
from sentence_transformers import SentenceTransformer



model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class Embedder:
    def __init__(self, model_name):
        self.model = model

    def embed_documents(self, documents):
        texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        embeddings = [self.model.encode(text, show_progress_bar=True).tolist() for text in texts]
        return embeddings

    def embed_query(self, query):
        embedding = self.model.encode(query, show_progress_bar=False).tolist()
        return embedding



def create_chain():
    client = chromadb.HttpClient(host="127.0.0.1",settings=Settings(allow_reset=True))

    embeddings = Embedder(model)
    db = Chroma(client=client, embedding_function=embeddings)
    # retv = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    # retv = db.as_retriever(search_kwargs={"k": 8})
    retv = db.as_retriever(search_type="similarity" ,search_kwargs={"k": 10})

    llm= ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key="YOUR_API_KEY",
    max_tokens=300,
    temperature=0.5,
    ) 
    memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, output_key='answer')

    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retv , memory=memory,
                                               return_source_documents=True)
    return qa


chain = create_chain()

def chat(user_message):
    bot_json = chain.invoke({"question": user_message})
    print(bot_json)
    return {"bot_response": bot_json}


if __name__ == "__main__":
    import streamlit as st
    st.title("Puranas Chatbot")
    col1 , col2 = st.columns([4,1])

    user_input = st.chat_input()
    with col1:
        col1.subheader("------Ask me a question about Puranas------")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if user_input:
            bot_response = chat(user_input)
            st.session_state.messages.append({"role" : "chatbot", "content" : bot_response})
            
            for message in st.session_state.messages:
                st.chat_message("user")
                st.write("Question: ", message['content']['bot_response']['question'])
                st.chat_message("assistant")
                st.write("Answer: ", message['content']['bot_response']['answer'])
            #with col2:
                st.chat_message("assistant")
                for doc in message['content']['bot_response']['source_documents']:
                    st.write("Reference: ", doc.metadata['source'] + "  \n"+ "-page->"+str(doc.metadata['page']))

