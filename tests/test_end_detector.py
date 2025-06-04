import os
import pytest

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from cogniweave.core.end_detector import ConversationEndDetector

load_dotenv()

skip_if_no_openai = pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Need OPENAI_API_KEY to access openai provider",
)

@skip_if_no_openai
def test_end_detector_should_detect_end() -> None:
    """Use explicit end message, expect to return True."""
    detector = ConversationEndDetector()
    # Explicit end message
    messages = [HumanMessage(content="好的，謝謝你的幫助，再見！")]

    result = detector.invoke({"messages": messages})
    assert result.end is True, "The detector should return a BaseModel, and result.end should be True to indicate the conversation is over"

@skip_if_no_openai
def test_end_detector_should_not_detect_end() -> None:
    """Use normal question message, expect to return False."""
    detector = ConversationEndDetector()
    # Continue conversation
    messages = [HumanMessage(content="你可以介紹一下巴黎有哪些必去的景點嗎？")]

    result = detector.invoke({"messages": messages})
    assert result.end is False, "The detector should return a BaseModel, and result.end should be False to indicate the conversation is not over" 