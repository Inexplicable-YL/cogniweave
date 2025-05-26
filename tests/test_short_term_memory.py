"""測試 Short-term Memory 功能的腳本"""
import asyncio
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.memory.updater import ShortTermMemoryChatUpdater
from src.prompts.generator import ShortMemoryPromptTemplate

load_dotenv()

def create_test_conversation() -> dict[str, Any]:
    """創建測試用的對話資料"""
    return {
        "name": "言靈",
        "history": [
            HumanMessage("我最近在學習 React，遇到了 useEffect 的問題。"),
            AIMessage("useEffect 確實是 React 中比較複雜的概念。你遇到了什麼具體問題？"),
            HumanMessage("就是不知道什麼時候該用依賴數組。"),
            AIMessage(
                "依賴數組決定了 effect 何時重新執行。空數組表示只在組件掛載時執行一次，"
                "有依賴項則在依賴項變化時重新執行。"
            ),
            HumanMessage("原來如此！謝謝你的解釋，我明白了。"),
            AIMessage("不客氣！如果還有其他 React 相關的問題，隨時可以問我。"),
        ],
        "timestamp": datetime.now().timestamp(),
    }


def create_art_conversation() -> dict[str, Any]:
    """創建另一個測試對話（繪畫主題）"""
    return {
        "name": "小夢",
        "history": [
            HumanMessage("我剛畫了一張初音未來，給你看看。"),
            AIMessage("哇，好漂亮的畫！你畫得真細膩，特別是頭髮的光澤處理得很好。"),
            HumanMessage("謝謝，我還擔心你不喜歡呢。"),
            AIMessage("我真的很喜歡！你的畫技太棒了，能看出你在細節上花了很多心思。"),
            HumanMessage("嘿嘿，我練習了好久呢。"),
        ],
        "timestamp": datetime.now().timestamp(),
    }


def print_memory_result(result: ShortMemoryPromptTemplate, title: str) -> None:
    """格式化輸出記憶結果"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"時間戳記: {result.timestamp}")
    print(f"標籤: {result.topic_tags}")
    print(f"\n格式化後的記憶:")
    print(result.format())
    print(f"{'='*50}\n")


def check_api_key():
    """檢查 API key 是否設置"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 錯誤：未找到 OPENAI_API_KEY")
        print("請在專案根目錄創建 .env 文件，並添加：")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    print(f"✓ 找到 API Key: {api_key[:10]}...")
    return True


async def test_sync_memory():
    """測試同步版本的短期記憶生成"""
    print("開始測試同步版本...")
    
    # 初始化 updater
    updater = ShortTermMemoryChatUpdater(lang="zh")
    
    # 測試第一個對話
    conv1 = create_test_conversation()
    result1 = updater.invoke(conv1)
    print_memory_result(result1, "對話 1: React 學習")
    
    # 測試第二個對話
    conv2 = create_art_conversation()
    result2 = updater.invoke(conv2)
    print_memory_result(result2, "對話 2: 繪畫分享")
    
    return result1, result2


async def test_async_memory():
    """測試異步版本的短期記憶生成"""
    print("\n開始測試異步版本...")
    
    # 初始化 updater
    updater = ShortTermMemoryChatUpdater(lang="zh")
    
    # 測試異步調用
    conv = create_test_conversation()
    result = await updater.ainvoke(conv)
    
    print(f"\n異步結果:")
    print(f"摘要: {result.chat_summary}")
    print(f"標籤: {result.topic_tags}")
    
    return result


async def test_english_version():
    """測試英文版本"""
    print("\n開始測試英文版本...")
    
    # 創建英文對話
    english_conv = {
        "name": "James",
        "history": [
            HumanMessage("I just finished a drawing of Hatsune Miku. Want to see?"),
            AIMessage("Wow, it's so beautiful! Your lines are really delicate."),
            HumanMessage("Thanks, I was worried you might not like it."),
            AIMessage("I really love it—your skill is amazing."),
        ],
        "timestamp": datetime.now().timestamp(),
    }
    
    # 使用英文版 updater
    updater = ShortTermMemoryChatUpdater(lang="en")
    result = updater.invoke(english_conv)
    
    print_memory_result(result, "English Conversation: Art Sharing")
    
    return result


def test_error_handling():
    """測試錯誤處理"""
    print("\n測試錯誤處理...")
    
    updater = ShortTermMemoryChatUpdater(lang="zh")
    
    # 測試無效輸入
    try:
        invalid_input = {
            "name": "測試用戶",
            "history": "這不是一個列表",  # 錯誤：應該是列表
            "timestamp": datetime.now().timestamp(),
        }
        updater.invoke(invalid_input)
        print("❌ 應該拋出錯誤但沒有")
    except TypeError as e:
        print(f"✓ 成功捕獲類型錯誤: {e}")
    except Exception as e:
        print(f"❌ 捕獲了意外的錯誤類型: {type(e).__name__}: {e}")
    
    # 測試缺少必要欄位
    try:
        missing_field = {
            "name": "測試用戶",
            # 缺少 history
            "timestamp": datetime.now().timestamp(),
        }
        updater.invoke(missing_field)
        print("❌ 應該拋出錯誤但沒有")
    except TypeError as e:
        print(f"✓ 成功捕獲欄位缺失錯誤: {e}")
    except Exception as e:
        print(f"❌ 捕獲了意外的錯誤類型: {type(e).__name__}: {e}")
    
    print("錯誤處理測試完成")


async def main():
    """主測試函數"""
    print("=" * 70)
    print("Short-term Memory 測試腳本")
    print("=" * 70)
    
    # 檢查 API key
    if not check_api_key():
        return
    
    # 設置環境變數（如果需要自定義模型）
    if not os.getenv("SHORT_MEMORY_MODEL"):
        os.environ["SHORT_MEMORY_MODEL"] = "openai/gpt-4.1-mini"
        print(f"✓ 設置預設模型: {os.environ['SHORT_MEMORY_MODEL']}")
    
    try:
        # 執行各項測試
        sync_results = await test_sync_memory()
        async_result = await test_async_memory()
        english_result = await test_english_version()
        
        # 錯誤處理測試單獨包裝
        try:
            test_error_handling()
        except Exception as e:
            print(f"錯誤處理測試失敗: {e}")
        
        print("\n測試完成！")
        print(f"共生成了 {len(sync_results) + 1 + 1} 個記憶實例")
        
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
        print("請檢查 API key 是否正確，或網路連接是否正常。")


if __name__ == "__main__":
    asyncio.run(main()) 