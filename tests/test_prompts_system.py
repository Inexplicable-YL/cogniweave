from datetime import UTC, datetime

from cogniweave.core.prompts.short_memory import ShortMemoryPromptTemplate
from cogniweave.prompts import MessageSegmentsPlaceholder, RichHumanMessagePromptTemplate


def _create_memory(ts: datetime, summary: str) -> ShortMemoryPromptTemplate:
    return ShortMemoryPromptTemplate.from_template(
        timestamp=ts,
        chat_summary=summary,
        topic_tags=[],
    )


def test_message_segments_placeholder() -> None:
    mem1 = _create_memory(datetime(2024, 1, 1, 12, 0, tzinfo=UTC), "hello")
    mem2 = _create_memory(datetime(2024, 1, 2, 13, 30, tzinfo=UTC), "world")

    placeholder = MessageSegmentsPlaceholder(variable_name="memories")
    template = RichHumanMessagePromptTemplate.from_template(
        ["prefix\n", placeholder, mem2, "suffix"]
    )

    test_timestamp = datetime(2024, 1, 2, 14, 0, tzinfo=UTC)
    message = template.format(memories=[mem1, mem2], timestamp=test_timestamp)
    expected = (
        "prefix\n"
        + mem1.format(timestamp=test_timestamp)
        + mem2.format(timestamp=test_timestamp) * 2
        + "suffix"
    )

    assert message.content == [{"type": "text", "text": expected}]


async def test_message_segments_placeholder_async() -> None:
    mem1 = _create_memory(datetime(2024, 1, 3, 10, 0, tzinfo=UTC), "foo")
    mem2 = _create_memory(datetime(2024, 1, 4, 11, 0, tzinfo=UTC), "bar")

    placeholder = MessageSegmentsPlaceholder(variable_name="memories")
    template = RichHumanMessagePromptTemplate.from_template(
        ["prefix\n", placeholder, mem2, "suffix"]
    )

    test_timestamp = datetime(2024, 1, 2, 14, 0, tzinfo=UTC)
    message = await template.aformat(memories=[mem1, mem2], timestamp=test_timestamp)
    expected = (
        "prefix\n"
        + await mem1.aformat(timestamp=test_timestamp)
        + await mem2.aformat(timestamp=test_timestamp) * 2
        + "suffix"
    )

    assert message.content == [{"type": "text", "text": expected}]


if __name__ == "__main__":
    test_message_segments_placeholder()
