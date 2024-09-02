import time
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

class LanguageModelProcessor:
    def format_docs(self,docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    def __init__(self):
        self.llm = ChatGroq(temperature=0.6, model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("INFERENCE_API_KEY"), model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
        self.db = Chroma(collection_name="product_collection", persist_directory="E:/JK Voice Assistant/chromadb_langchain", embedding_function=self.embeddings)

        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        system_prompt = system_prompt+ "{context}"
        # self.prompt = ChatPromptTemplate.from_messages([
        #     SystemMessagePromptTemplate.from_template(system_prompt),
        #     HumanMessagePromptTemplate.from_template("{text}")
        # ])

        # self.prompt = ChatPromptTemplate.from_template(system_prompt)
        # self.prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", system_prompt),
        #         ("human", "{input}"),
        #     ]
        # )
        self.retriever = self.db.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)
        # self.conversation = LLMChain(
        #     llm=self.llm,
        #     prompt=self.prompt,
        #     memory=self.memory
        # )
        
        # self.chain =(RunnablePassthrough.assign(context = lambda input: self.format_docs(input["context"]))
        #         |self.conversation
        #         |StrOutputParser()
        #         )
        

        # self.question_answer_chain = create_stuff_documents_chain(self.conversation, self.prompt)

        # self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)

        # self.rag_chain =( 
        #                 {"context": self.retriever | self.format_docs, "text": RunnablePassthrough()} 
        #                  | self.conversation
        #                  |StrOutputParser()
        #                 )
    
    def process(self, text):
        self.memory.chat_memory.add_user_message(text)
        start_time = time.time()
        chat_history=[]
        # rag_response = self.db.similarity_search_by_vector_with_relevance_scores(
        #     embedding=self.embeddings.embed_query(text), k=2
        # )

        # rag_response_text = "\n".join([doc[0].page_content for doc in rag_response])
        # combined_text = f"User input: {text}\nDatabase response: {rag_response_text}"

        # docs = self.db.similarity_search(text)
        # self.chain.invoke({"context":docs, "question":text})
        response = self.rag_chain.invoke({"input":text, "chat_history":chat_history})

        end_time = time.time()
        self.memory.chat_memory.add_ai_message(response['answer'])
        elapsed_time = int((end_time - start_time) * 1000)
        return response['answer'], elapsed_time

# Testing the above class:
# if __name__ == "__main__":
#     processor = LanguageModelProcessor()

#     # Define a test input
#     test_input = "what is its price?"

#     response, elapsed_time = processor.process(test_input)

#     # Print the response and elapsed time
#     print(f"Response: {response}")
#     print(f"Elapsed time: {elapsed_time} ms")

# # language_model.py
# import time
# import os
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import LLMChain
# from langchain.prompts import (
#     ChatPromptTemplate,
#     MessagesPlaceholder,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from langchain_chroma import Chroma


# load_dotenv()

# class LanguageModelProcessor:
#     def __init__(self):

#         self.llm = ChatGroq(temperature=0.6, model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
#         self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#         self.embeddings = HuggingFaceInferenceAPIEmbeddings(
#             api_key=os.getenv("inference_api_key"), model_name="sentence-transformers/all-MiniLM-l6-v2"
#         )
#         # print(self.embeddings)

#         self.db = Chroma(collection_name="product_collection", persist_directory="E:\JK Voice Assistant\chromadb_langchain", embedding_function=self.embeddings)
#         # print(self.db.get())

#         with open('system_prompt.txt', 'r') as file:
#             system_prompt = file.read().strip()
        

#         self.prompt = ChatPromptTemplate.from_messages([
#             SystemMessagePromptTemplate.from_template("Here is the response from the product database {text}. Use this as a primary source of information.If not present in database, never mention anything about database to the user.Search your own database and don't respond no product found "+system_prompt),
#             HumanMessagePromptTemplate.from_template("{text}")
#         ])


#         self.conversation = LLMChain(
#             llm=self.llm,
#             prompt=self.prompt,
#             memory=self.memory
#         )

#     def process(self, text):
#         self.memory.chat_memory.add_user_message(text)
#         start_time = time.time()

#         rag_response = self.db.similarity_search_by_vector_with_relevance_scores(
#             embedding= self.embeddings.embed_query(text), k=2
#         )

#         # print(rag_response)

#         rag_response_text = "\n".join([doc.page_content for doc in rag_response])

#         combined_text = text + "||||" + rag_response_text

#         response = self.conversation.invoke({"text": combined_text})

#         # print(response, rag_response)
#         # Ensure the keys align with the prompt template

#         # Debugging prints
#         # print("Invocation payload:", input_data)

#         # Invoke the conversation chain
#         # response = self.conversation.invoke()

#         end_time = time.time()
#         self.memory.chat_memory.add_ai_message(response['text'])
#         elapsed_time = int((end_time - start_time) * 1000)
#         return response['text'], elapsed_time

# #Testing the above class:
# if __name__ == "__main__":
#     processor = LanguageModelProcessor()

#     # Define a test input
#     test_input = "Suggest me a tshirt."

#     response, elapsed_time = processor.process(test_input)

#     # Print the response and elapsed time
#     print(f"Response: {response}")
#     print(f"Elapsed time: {elapsed_time} ms")