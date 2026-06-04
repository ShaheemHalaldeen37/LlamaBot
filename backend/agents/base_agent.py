from abc import ABC, abstractmethod
import sys, os

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm_provider import build_llm

class BaseAgent(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

        load_dotenv()
        self.llm = build_llm()

    @abstractmethod
    def run(self, input: str) -> str:
        pass

    def invoke(self, input: str) -> str:
        return self.llm.invoke(input)
