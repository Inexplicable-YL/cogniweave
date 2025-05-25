from typing import Any

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from src.prompts import (
    MessageSegmentsPlaceholder,
    MessageSegmentsPromptTemplate,
    RichSystemMessagePromptTemplate,
)

prompt = ChatPromptTemplate.from_messages(
    [
        RichSystemMessagePromptTemplate.from_template(
            [
                "这是前面的文本",
                MessageSegmentsPlaceholder(variable_name="prompts"),
                "这是后面的文本",
            ]
        ),
        HumanMessagePromptTemplate.from_template("这是用户发的话"),
    ]
)


# langchain的伪函数
def invoke(input: dict[Any, Any]) -> str: ...


invoke({"prompts": [MessageSegmentsPromptTemplate()]})

"""
最终呈现的system，一个str
```
这是前面的文本
MessageSegmentsPromptTemplate里面格式化产生我文本
这是后面的文本
```
"""
