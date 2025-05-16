import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
import openai

def generate_map_url(latitude: float, longitude: float) -> str:
    """Generate a map URL for the given coordinates."""
    return f"https://baxtel.com/map?lat={latitude}&lng={longitude}&distance=173932.67940883202"

class DatacenterRAG:
    def __init__(self, api_key=None):
        print(f"[rag_system.py] DatacenterRAG __init__ called.")
        self.openai_token = api_key
        
        if not self.openai_token:
            print("[rag_system.py] CRITICAL ERROR: OpenAI API key not provided to DatacenterRAG constructor.")
            raise ValueError("OpenAI API key not provided to DatacenterRAG constructor")
        
        print(f"[rag_system.py] Received API Key in RAG: {'sk-...' + self.openai_token[-4:] if len(self.openai_token) > 7 and self.openai_token.startswith('sk-') else 'Key format potentially incorrect'}")

        # Ensure API key is available to OpenAI client and as environment variable for LangChain fallbacks
        openai.api_key = self.openai_token
        os.environ["OPENAI_API_KEY"] = self.openai_token
        print(f"[rag_system.py] Set openai.api_key and os.environ['OPENAI_API_KEY'].")

        # Load the data with error handling
        try:
            print("[rag_system.py] Attempting to load data files...")
            # First try project root directory paths
            data_path = 'data/'
            prompts_path = 'prompts/'
            try:
                self.existing_dcs = pd.read_csv(os.path.join(data_path, 'us_datacenters.csv'), on_bad_lines='skip')
                self.possible_locations = pd.read_csv(os.path.join(data_path, 'us_possible_locations.csv'), on_bad_lines='skip')
                with open(os.path.join(prompts_path, 'datacenter_site_selector.txt'), 'r') as f:
                    self.site_selector_prompt = f.read()
                print("[rag_system.py] Loaded data from default 'data/' and 'prompts/' paths.")
            except FileNotFoundError:
                print("[rag_system.py] Default file paths not found. Trying alternative paths for Hugging Face Space...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_path_alt = os.path.join(current_dir, 'data/')
                prompts_path_alt = os.path.join(current_dir, 'prompts/')
                
                self.existing_dcs = pd.read_csv(os.path.join(data_path_alt, 'us_datacenters.csv'), on_bad_lines='skip')
                self.possible_locations = pd.read_csv(os.path.join(data_path_alt, 'us_possible_locations.csv'), on_bad_lines='skip')
                with open(os.path.join(prompts_path_alt, 'datacenter_site_selector.txt'), 'r') as f:
                    self.site_selector_prompt = f.read()
                print("[rag_system.py] Loaded data from alternative paths relative to rag_system.py.")
            print("[rag_system.py] Data files loaded successfully.")
        except Exception as e:
            print(f"[rag_system.py] Error loading data files: {type(e).__name__} - {str(e)}")
            raise
        
        self.qa_prompt = PromptTemplate.from_template("""
        You are a datacenter site selection expert assistant. Use ONLY the following context to answer the user's question. 
        Do NOT make up any information that is not present in the context. If you don't know, say you don't know.
        When mentioning locations, ALWAYS include the precise coordinates, land price, electricity cost, and any notes mentioned in the context.
        Include the map URL for each location you mention.
        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        Answer:""")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function=len, separators=["\n\n", "\n", " ", ""]
        )
        
        try:
            print("[rag_system.py] Initializing OpenAIEmbeddings...")
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_token,
                model="text-embedding-ada-002"
            )
            print("[rag_system.py] OpenAIEmbeddings initialized.")
        except Exception as e:
            print(f"[rag_system.py] Error initializing OpenAIEmbeddings: {type(e).__name__} - {str(e)}")
            raise

        self.vector_store = None
        try:
            print("[rag_system.py] Initializing vector store (calling self.initialize_vector_store)...")
            self.initialize_vector_store() 
            print("[rag_system.py] Vector store initialized successfully after call to self.initialize_vector_store.")
        except Exception as e:
            print(f"[rag_system.py] Error during self.initialize_vector_store: {type(e).__name__} - {str(e)}")
            raise

        try:
            print(f"[rag_system.py] Initializing ChatOpenAI with model 'gpt-4o-mini'...")
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=2000,
                openai_api_key=self.openai_token
            )
            print("[rag_system.py] ChatOpenAI initialized.")
        except Exception as e:
            print(f"[rag_system.py] Error initializing ChatOpenAI: {type(e).__name__} - {str(e)}")
            raise
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True,
            return_generated_question=True,
            output_key="answer",
            max_tokens_limit=4000,
        )
        
        self.message_history = ChatMessageHistory()
        self.chain_with_history = RunnableWithMessageHistory(
            self.qa_chain,
            lambda session_id: self.message_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        print("[rag_system.py] DatacenterRAG initialization complete.")

    def initialize_vector_store(self):
        print("[rag_system.py] self.initialize_vector_store called.")
        try:
            existing_dc_texts = self.existing_dcs.apply(
                lambda x: f"Existing Datacenter: {x['location_name']} at coordinates ({x['latitude']}, {x['longitude']}). "
                f"Land price: ${x['land_price']} per acre. Electricity cost: ${x['electricity']} per kWh. "
                f"Notes: {x['notes']}. "
                f"Map URL: {generate_map_url(x['latitude'], x['longitude'])}",
                axis=1
            ).tolist()
            
            possible_loc_texts = self.possible_locations.apply(
                lambda x: f"Potential Location: {x['location name']} at coordinates ({x['latitude']}, {x['longitude']}). "
                f"Land price: ${x['land price']} per acre. Electricity cost: ${x['electricity']} per kWh. "
                f"Notes: {x['notes']}. "
                f"Map URL: {generate_map_url(x['latitude'], x['longitude'])}",
                axis=1
            ).tolist()
            
            all_texts = [self.site_selector_prompt] + existing_dc_texts + possible_loc_texts
            print(f"[rag_system.py] Number of texts to vectorize: {len(all_texts)}")
            if len(all_texts) > 1:
                 print(f"[rag_system.py] Sample text for vectorization (first 100 chars): {all_texts[1][:100]}...")
            
            texts = self.text_splitter.create_documents(all_texts)
            print(f"[rag_system.py] Texts split into {len(texts)} documents for FAISS.")
            
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            print(f"[rag_system.py] FAISS vector store created with {len(texts)} documents.")
        except Exception as e:
            print(f"[rag_system.py] Error in initialize_vector_store: {type(e).__name__} - {str(e)}")
            raise
    
    def query(self, question: str) -> str:
        print(f"[rag_system.py] Query received: {question[:100]}...")
        try:
            if not self.message_history.messages:
                print("[rag_system.py] New conversation: Adding initial context for site selection.")
            
            result = self.chain_with_history.invoke(
                {"question": question},
                config={"configurable": {"session_id": "user_session"}}
            )
            
            if isinstance(result, dict) and "answer" in result:
                if "source_documents" in result and result["source_documents"]:
                    print(f"[rag_system.py] Query answered. Number of source documents: {len(result['source_documents'])}")
                else:
                    print("[rag_system.py] Query answered. No source documents returned or key missing.")
                return result["answer"]
            elif isinstance(result, str):
                print("[rag_system.py] Query returned a string directly.")
                return result
            else:
                print(f"[rag_system.py] Query returned unexpected result type: {type(result)}. Converting to string.")
                return str(result)
                
        except Exception as e:
            print(f"[rag_system.py] Error during query processing: {type(e).__name__} - {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please check the logs for details." 