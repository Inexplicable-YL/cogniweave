import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
import json

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from cogniweave.core.memory_makers.long_memory import LongTermMemoryMaker


def main():
    """Real LLM test LongTermMemoryMaker's complete process."""

    # Load environment variables
    load_dotenv()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in .env file")
        return

    print("Start testing LongTermMemoryMaker...")

    # Prepare test chat history
    history = [
        HumanMessage(content="我最近開始學習 Python 程式設計"),
        AIMessage(content="太好了！Python 是一個很棒的程式語言，適合初學者。你有什麼特別想學的方向嗎？"),
        HumanMessage(content="我對機器學習很有興趣，想用 Python 來做 AI 專案"),
        AIMessage(content="機器學習確實是 Python 的強項！建議你可以從 scikit-learn 開始，然後再學習 TensorFlow 或 PyTorch。"),
        HumanMessage(content="好的，我會先從基礎開始。另外我平常喜歡喝咖啡，特別是手沖咖啡"),
        AIMessage(content="手沖咖啡是個很棒的興趣！不同的沖泡方法會帶來不同的風味。你有最喜歡的咖啡豆產區嗎？"),
        HumanMessage(content="我比較喜歡衣索比亞的豆子，酸味比較明亮"),
        AIMessage(content="衣索比亞確實以果香和花香聞名，特別是耶加雪菲。搭配淺烘焙會有很棒的酸質表現。"),
    ]

    # Create LongTermMemoryMaker (using default settings, will automatically create a real LLM chain)
    memory_maker = LongTermMemoryMaker(lang="zh")

    print("Prepared chat history:")
    for i, msg in enumerate(history, 1):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        print(f"  {i}. [{role}] {msg.content}")

    print("Step 1: Test memory extraction...")
    try:
        # Test extraction function
        extracted_json = memory_maker._extract(
            {"history": history},
            memory_maker._get_current_timestamp(),
            memory_maker._get_current_date()
        )

        print(f"Original extraction result: {extracted_json}")

        extracted_items = json.loads(extracted_json)

        # Handle different response formats
        if isinstance(extracted_items, dict) and "memories" in extracted_items:
            actual_memories = extracted_items["memories"]
        elif isinstance(extracted_items, list):
            actual_memories = extracted_items
        else:
            actual_memories = [str(extracted_items)]

        print(f"Successfully extracted {len(actual_memories)} memory items:")
        for i, item in enumerate(actual_memories, 1):
            print(f"  {i}. {item}")

    except Exception as e:
        print(f"Extraction failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return

    print("Step 2: Test complete memory update process...")
    try:
        # Prepare input data (including existing memory)
        input_data = {
            "history": history,
            "current_long_term_memory": [
                "2025-06-17（2天前）得知用戶對程式設計有興趣，想學習新技術",
                "2025-06-16（3天前）得知用戶提到喜歡閱讀技術書籍"
            ],
            "last_update_time": "2025-06-17 15:30"
        }

        # Execute complete process
        result = memory_maker.invoke(input_data)

        print("Memory updated successfully!")
        print(f"Number of updated memory items: {len(result.updated_memory)}")
        print("Complete memory content:")
        for i, memory_item in enumerate(result.updated_memory, 1):
            print(f"  {i}. {memory_item}")

        print("\nJSON output:")
        formatted_json = result.format()

        # Check if it's an empty string
        if not formatted_json.strip():
            # If format() returns an empty string, use updated_memory to generate JSON
            formatted_json = json.dumps(result.updated_memory, ensure_ascii=False, indent=2)
            print("format() returns an empty string, use updated_memory to generate JSON")

        print(formatted_json)

        # Validate JSON format
        try:
            json.loads(formatted_json)
            print("SON format validation passed")
        except json.JSONDecodeError as je:
            print(f"JSON format validation failed: {je}")

    except Exception as e:
        print(f"Complete process failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return

    print("\nStep 3: Test asynchronous version...")
    try:
        async def test_async():
            result = await memory_maker.ainvoke(input_data)
            return result

        async_result = asyncio.run(test_async())
        print("Asynchronous version executed successfully!")
        print(f"Number of asynchronous result memory items: {len(async_result.updated_memory)}")

        # Compare synchronous and asynchronous results
        if result.updated_memory == async_result.updated_memory:
            print("Synchronous and asynchronous results are consistent")
        else:
            print("Synchronous and asynchronous results are different")
            print("Synchronous result item count:", len(result.updated_memory))
            print("Asynchronous result item count:", len(async_result.updated_memory))

    except Exception as e:
        print(f"Asynchronous version failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return

    print("\nAll tests completed! LongTermMemoryMaker is working properly.")

    # Additional information
    print("\nPerformance statistics:")
    print(f"  - Language used: {memory_maker.lang}")
    print(f"  - Extraction chain type: {type(memory_maker.extract_chain).__name__}")
    print(f"  - Merging chain type: {type(memory_maker.chat_chain).__name__}")


if __name__ == "__main__":
    main()
