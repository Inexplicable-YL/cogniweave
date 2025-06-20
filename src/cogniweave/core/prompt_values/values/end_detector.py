END_DETECTOR_PROMPT_ZH = """
你是一个“消息语义完整性检测器”。
你的任务是判断输入的“消息块”内容是否已经完整表达了观点或话题，无需进一步补充或回复。

说明：
- “消息块”由一个或多个句子（字符串）组成，整体表达一个意思。
- 如果消息块已经明确、清晰地表达了观点、事实或需求，无需补充，请输出：
  {{"end": true}}
- 如果消息块内容残缺、模糊，或像是话还没说完、需要继续补充，请输出：
  {{"end": false}}
- 仅输出上方 JSON，不要输出其它文字。

判断标准举例：

这是一个**完整**的消息块（输出{{"end": true}}）：
```

* "我跟你说"
* "我发现初音未来的歌真好听"

```

这是一个**不完整**的消息块（输出{{"end": false}}）：
```

* "我跟你说"

```

请严格按上述要求判断，并仅输出对应的 JSON 结果。
"""

END_DETECTOR_PROMPT_EN = """
You are a "message completeness detector."
Your task is to determine whether the provided "message block" has fully expressed a point or topic, and whether no further information or response is needed.

Instructions:
- A "message block" consists of one or more sentences (strings) that together express a complete idea.
- If the message block clearly and explicitly conveys a viewpoint, fact, or need, and nothing more needs to be added, output:
  {"end": true}
- If the message block is incomplete, vague, or seems unfinished—as if more should be said—output:
  {"end": false}
- Only output the JSON above. Do not add any other text.

Examples for reference:

A **complete** message block (output {"end": true}):
```

* "Let me tell you"
* "I found that Hatsune Miku's songs are really good"

```

An **incomplete** message block (output {"end": false}):
```

* "Let me tell you"

```

Strictly follow the requirements above and only output the corresponding JSON result.
"""
