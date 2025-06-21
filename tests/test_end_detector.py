import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from cogniweave.core.end_detector import EndDetector

load_dotenv()

skip_if_no_openai = pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Need OPENAI_API_KEY to access openai provider",
)


@skip_if_no_openai
def test_end_detector_should_detect_end() -> None:
    """Use explicit end message, expect to return True."""
    detector = EndDetector()
    # Explicit end message
    messages = [
        HumanMessage(content="今天领导大赦我们"),
        HumanMessage(content="今天下午不用上班，可以好好休息一下"),
    ]

    result = detector.invoke({"messages": messages})
    assert result is True, (
        "The detector should return a BaseModel, and result.end should be True to indicate the conversation is over"
    )


@skip_if_no_openai
def test_end_detector_should_not_detect_end() -> None:
    """Use normal question message, expect to return False."""
    detector = EndDetector()
    # Continue conversation
    messages = [
        HumanMessage(content="我今天遇到一件很奇怪的事情"),
        HumanMessage(content="你猜怎么着"),
    ]

    result = detector.invoke({"messages": messages})
    assert result is False, (
        "The detector should return a BaseModel, and result.end should be False to indicate the conversation is not over"
    )
