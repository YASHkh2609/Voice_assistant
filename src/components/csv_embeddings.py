import numpy as np
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from uuid import uuid4


load_dotenv()

class CSVEmbeddingProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        # self.embeddings = None
        # self.embedding_model = SentenceTransformer(embedding_model)

    def load_csv(self):
        """Load CSV data into a pandas DataFrame."""
        try:
            self.data = pd.read_csv(self.csv_path,lineterminator='\n')
            print(f"CSV data loaded successfully from {self.csv_path}")
        except Exception as e:
            print(f"Error loading CSV file: {e}")

    def preprocess_data(self):
        """
        Preprocess the text data from the relevant columns: 'title', 'description', and 'highlights'.
        Combines these columns into a single text entry per row.
        """


        if self.data is None:
            print("Data not loaded.")
            return []
        # Combine title, description, and highlights into a single text for each product

        descriptions = []
        i=1
        for _, row in self.data.iterrows():
            description = Document(
                page_content=(f"The category of this product is {row['category_1']} and subcategories are "
                f"{row['category_2']} and {row['category_3']}. The title of the product is '{row['title']}' "
                f"with a product rating of {row['product_rating']} and a selling price of {row['selling_price']} "
                f"(MRP {row['mrp']}). The product is sold by {row['seller_name']} with a seller rating of "
                f"{row['seller_rating']}. Product description: {row['description']}. Highlights include: "
                f"{row['highlights']}."),
                id=i
            )
            i+=1
            descriptions.append(description)
        return descriptions

    # def create_embeddings(self, texts):
    #     """
    #     Generate embeddings for the list of texts using the specified embedding model.
    #     """
    #     if not texts:
    #         print("No texts provided for embedding creation.")
    #         return

    #     print("Generating embeddings...")
    #     self.embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
    #     print(f"Generated {len(self.embeddings)} embeddings.")

    # def store_embeddings(self, index_file='embeddings.index'):
    #     """
    #     Store embeddings using FAISS for efficient similarity search.
    #     """
    #     if self.embeddings is None:
    #         print("No embeddings to store.")
    #         return

    #     # Convert embeddings to a numpy array and create a FAISS index
    #     embedding_matrix = np.array(self.embeddings).astype('float32')
    #     index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance index
    #     index.add(embedding_matrix)

    #     # Save the index to a file
    #     faiss.write_index(index, index_file)
    #     print(f"Embeddings stored in {index_file}.")

    def run(self):
        """
        Complete pipeline: Load CSV, preprocess data, generate embeddings, and store them.
        """
        self.load_csv()
        texts = self.preprocess_data()
        return texts
        # self.create_embeddings(texts)
        # self.store_embeddings()

# Example usage
processor = CSVEmbeddingProcessor(csv_path='cleaned_data.csv')
documents = processor.run()
print(documents)
print(documents[0])
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("inference_api_key"), model_name="sentence-transformers/all-MiniLM-l6-v2"
)

vector_store = Chroma(
    collection_name="product_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
