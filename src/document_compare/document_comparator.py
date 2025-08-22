import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception_archieve import DocumentPortalException
from model.models import *
from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY["document_comparison"]
        self.chain = self.prompt | self.llm | self.parser  
        self.log.info("DocumentComparatorLLM initialized with model and parser.")

    def compare_documents(self,combined_docs: str) -> pd.DataFrame:
        """
        Compares two documents and returns a structured comparison.
        """
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions()
            }
            self.log.info("Starting document comparison", inputs=inputs)
            response = self.chain.invoke(inputs)
            self.log.info("Document comparison completed", response=response)
            return self._format_response(response)

            
        except Exception as e:
            self.log.error(f"Error in compare_documents: {e}")
            raise DocumentPortalException("An error occurred while comparing documents.",sys)
    
    def _format_response(self,response_parsed: list[dict]) -> pd.DataFrame: #type: ignore
        """
        Formats the response from the LLM into a structured format.
        """
        try:
            df = pd.DataFrame(response_parsed)
            self.log.info("Response formatted into DataFrame", dataframe=df)
            return df
        except Exception as e:
            self.log.error("Error formatting response into DataFrame", error=str(e))
            raise DocumentPortalException("Error formatting response",sys)