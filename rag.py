import numpy as np
from rank_bm25 import BM25Okapi 
import google.genai as genai
from google.genai.errors import APIError
from google.genai import types
from sentence_transformers import SentenceTransformer

RETRIEVER_MODEL = 'BM25' 
GENERATOR_MODEL = 'gemini-2.5-flash'

class DiReCTRAG:
    def __init__(self, data_list, retriever_model_name=RETRIEVER_MODEL, generator_model_name=GENERATOR_MODEL):
        self.data_list = data_list
        self.documents = []
        self.tokenized_corpus = None 
        self.retriever = None 
        
        print("Loading SentenceTransformer model and tokenizer...")
        try:
            self.st_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
            self.tokenizer = self.st_model.tokenizer
            print("Tokenizer loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model/tokenizer: {e}")
        
        try:
            self.client = genai.Client()
        except Exception as e:
            raise ValueError("Failed to initialize Gemini Client. Ensure the GEMINI_API_KEY environment variable is set correctly.") from e
            
        self.generator_model_name = generator_model_name

    def _tokenize(self, text: str):
        tokens = self.tokenizer.tokenize(text)

        tokens = [t.lower() for t in tokens if t.isalnum()]
        return tokens

    def _flatten_data(self):
        print("Flattening and preparing documents for indexing...")

        documents = []
        def extract_nested_structure(node, path=None):
            if path is None:
                path = []

            if isinstance(node, dict):
                for key, value in node.items():
                    new_path = path + [key]

                    if isinstance(value, dict) and len(value) == 0:
                        doc = f"**Diagnostic Path:** {' → '.join(new_path)}"
                        documents.append(doc)
                    else:
                        extract_nested_structure(value, new_path)

            elif isinstance(node, list):
                if len(node) == 0:
                    doc = f"**Diagnostic Category:** {' → '.join(path)}"
                    documents.append(doc)
                else:
                    for item in node:
                        extract_nested_structure(item, path)

        def extract_knowledge_dict(knowledge_dict):
            if isinstance(knowledge_dict, dict):
                for key, value in knowledge_dict.items():

                    if isinstance(value, str) and value.strip():
                        documents.append(
                            f"**Concept:** {key}\n**Knowledge:** {value.strip()}"
                        )

                    elif isinstance(value, dict):
                        extract_knowledge_dict(value)


        for record in self.data_list:
            if "diagnostic" in record:
                extract_nested_structure(record["diagnostic"])
            if "knowledge" in record:
                extract_knowledge_dict(record["knowledge"])
            for key, value in record.items():
            # Skip non-nested keys
                if key.startswith("input") or key in ["diagnostic", "knowledge"]:
                    continue

                if isinstance(value, dict):
                    extract_nested_structure({key: value})
            for i in range(1, 7):
                input_key = f"input{i}"
                if input_key in record:
                    text = record[input_key]

                    if isinstance(text, str) and text.strip():
                        documents.append(
                            f"**Clinical Note Segment ({input_key}):**\n{text.strip()}"
                        )
        self.documents = list(set(documents))

        print(f"Data flattened into {len(self.documents)} unique documents.")

    
    def _create_index(self):
        if not self.documents:
            self._flatten_data()

        print("Tokenizing corpus with nomic-embed-text-v1.5 tokenizer...")

        self.tokenized_corpus = [self._tokenize(doc) for doc in self.documents]

        self.retriever = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 index created with {len(self.documents)} documents.")


    def retrieve_documents(self, query: str, k: int = 5):
        """Performs a BM25 search."""
        if self.retriever is None:
            self._create_index()

        tokenized_query = self._tokenize(query)
        scores = self.retriever.get_scores(tokenized_query)
        top_k = np.argsort(scores)[::-1][:k]
        
        retrieved_results = []
        for idx in top_k:
            retrieved_results.append({
                "document": self.documents[idx],
                "distance": scores[idx] # BM25 score (higher is better)
            })
            
        return retrieved_results
    
    def _create_prompt(self, user_query: str, retrieved_docs: list) -> str:
    
        context = "\n---\n".join([r['document'] for r in retrieved_docs])
        
        prompt = f"""You are a specialized AI designed for **Academic Information Synthesis and Data Analysis** based solely on the provided source materials from a clinical dataset. Your goal is to synthesize the requested information for research purposes.
        
        **STRICT RULE:** You **MUST** use ONLY the information contained within the **Context** section. Do not provide diagnostic recommendations or clinical advice. If the context does not contain sufficient information, state that the relevant data is unavailable in the provided sources.

        **Academic Synthesis Request:** Based on the context, please synthesize the information needed to address the following query: {user_query}

        **Context (Retrieved Clinical Knowledge and Notes):**
        {context}

        **Synthesized Report:**
        """
        
        return prompt

    def generate_response(self, query: str, k: int = 5):
    
        retrieved_docs = self.retrieve_documents(query, k)
        full_prompt = self._create_prompt(query, retrieved_docs)
        
        print(f"Generating final response using {self.generator_model_name}...")
        
        try:
            response = self.client.models.generate_content(
                model=self.generator_model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=2048
                )
            )
            
            if response.text is not None and response.text.strip():
                final_answer = response.text.strip()
            else:
                final_answer = (
                    "**Error: Generation Blocked or Empty.** "
                    "The model did not return a valid response. "
                    "This may be due to safety filters blocking the content "
                )

        except APIError as e:
            final_answer = f"Error calling Gemini API: {e}"
        except Exception as e:
            final_answer = f"An unexpected error occurred during generation: {e}"
                    
        return final_answer, retrieved_docs