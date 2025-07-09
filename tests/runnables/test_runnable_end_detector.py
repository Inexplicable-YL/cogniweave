import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from test_runnable import RunnableForTest

from cogniweave.core.end_detector import EndDetector
from cogniweave.core.runnables.end_detector import RunnableWithEndDetector

load_dotenv()

skip_if_no_openai = pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Need OPENAI_API_KEY to access openai provider",
)


@skip_if_no_openai
def test_end_detector_should_detect_end() -> None:
    """Use explicit end message, expect to return True."""
    runnable = RunnableForTest()
    end_detector = EndDetector()
    detector = RunnableWithEndDetector(
        runnable,
        end_detector=end_detector,
        default="test_default",
        history_messages_key="history",
    )

    # Explicit end message
    input = HumanMessage(content="下午不用上班了，可以好好休息一下")
    history = [HumanMessage(content="今天领导大赦我们")]

    result = detector.invoke({"input": input, "history": history, "answer": "test_answer"})
    assert result.content == "test_answer", "The detector should return the answer"  # type: ignore


@skip_if_no_openai
async def test_end_detector_should_not_detect_end_async() -> None:
    """Use normal question message, expect to return False."""
    runnable = RunnableForTest()
    end_detector = EndDetector()
    detector = RunnableWithEndDetector(
        runnable,
        end_detector=end_detector,
        default="test_default",
        history_messages_key="history",
    )

    input = HumanMessage(content="你猜怎么着")
    history = [HumanMessage(content="我今天遇到一件很奇怪的事情")]

    result = await detector.ainvoke({"input": input, "history": history, "answer": "test_answer"})
    assert result == "test_default", "The detector should return the default"
