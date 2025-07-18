
import re
from typing import List


from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableParallel

from WeaviateManager import WeaviateManager

# =====================

# RAG Pipeline
class RAGPipeline:
    """Manages the RAG pipeline creation and execution."""
    
    def __init__(self, weaviate_manager: WeaviateManager, embeddings_model, llm_model):
        self.weaviate_manager = weaviate_manager
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        
    def create_pipeline(self):
        """Create the RAG pipeline."""
        def retriever(query: str) -> List[Document]:
            docs = self.weaviate_manager.search_documents_vector_only(query, limit=3)
            return docs

        def format_docs(docs):
            formatted_docs = []
            for i, doc in enumerate(docs):
                # Extract the question from metadata if available
                #original_question = doc.metadata.get("question", "")
                # print(f"\nDocument {i+1}")
                # print(f"Score: {getattr(doc.metadata, 'score', 'N/A')}")
                # print(f"Content: {doc.page_content[:500]}...")  # print only first 500 chars to keep it readable
                # print(f"Metadata: {doc.metadata}")

                context = doc.page_content
                formatted_docs.append(f"ABSTRACT CONTEXT{i+1}: {context}\n")
            
            return "\n\n".join(formatted_docs)
        template = """ Based on the medical abstracts below, provide an evidence-based answer. 


ABSTRACTS:
{context}

QUESTION:
{question}

Answer: """

        prompt = ChatPromptTemplate.from_template(template)
        print(f"prompt: {prompt}")
        def clean_response(response: str) -> str:
            """Clean up the response."""
            if "Response:" in response:
                response = response.split("Response:")[-1].strip()
            
            response = re.sub(r'Based on .*?PubMedQA[: ]', '', response, flags=re.IGNORECASE | re.DOTALL)
            response = re.sub(r'Based on .*?abstracts[: ]', '', response, flags=re.IGNORECASE | re.DOTALL)
            response = re.sub(r'PubMedQA information:', '', response, flags=re.IGNORECASE)
            response = re.sub(r'Human:', '', response, flags=re.IGNORECASE)
            response = re.sub(r'Assistant:', '', response, flags=re.IGNORECASE)
            response = re.sub(r'\n\s*\n', '\n', response)
            response = re.sub(r'^\s*[-*]\s*', '', response)
            
            return response.strip()

        # Create a runnable parallel for input transformation
        setup_and_retrieval = RunnableParallel(
            context=lambda x: format_docs(retriever(x)),
            question=RunnablePassthrough()
        )

        # Compose the chain
        chain = (
            setup_and_retrieval 
            | prompt 
            | self.llm_model 
            | StrOutputParser()
            | clean_response
        )
        return chain

# =====================