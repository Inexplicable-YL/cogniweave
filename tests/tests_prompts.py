from langchain_core.load import load
from src.prompts.generator import ShortMemoryPromptTemplate
from datetime import datetime

memory = ShortMemoryPromptTemplate.from_template(
    timestamp=datetime.now(),
    chat_summary="11111",
    topic_tags=["111","222"]
)
json_memory = memory.to_template_dict()
print(ShortMemoryPromptTemplate.load(json_memory))