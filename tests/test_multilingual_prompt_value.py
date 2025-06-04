import pytest

from cogniweave.prompt_values import MultilingualSystemPromptValue


def test_to_messages_returns_expected_content() -> None:
    prompt = MultilingualSystemPromptValue(zh="你好", en="hello")
    zh = list(prompt.to_messages(lang="zh"))
    en = list(prompt.to_messages(lang="en"))
    assert zh[0].content == "你好"
    assert en[0].content == "hello"


def test_to_messages_unsupported_language() -> None:
    prompt = MultilingualSystemPromptValue(zh="你好", en="hello")
    with pytest.raises(ValueError):
        list(prompt.to_messages(lang="jp"))
