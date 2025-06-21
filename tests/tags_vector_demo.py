from pydantic import BaseModel

from cogniweave.llms import OpenAIEmbeddings
from cogniweave.vector_stores.tags import TagsVector


# 定义测试用的 Pydantic 模型
class ContentItem(BaseModel):
    text: str
    category: str


# 创建两种类型的 TagsVector 实例
str_vector = TagsVector[str](
    folder_path="./.cache/str_cache",
    index_name="index-str",
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
    auto_save=True,
)

model_vector = TagsVector[ContentItem](
    folder_path="./.cache/model_cache",
    index_name="index-model",
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
    auto_save=True,
)

samples = [
    (["生日", "朋友", "惊喜"], "今天是我的生日，朋友们为我准备了惊喜派对。", {"category": "事件"}),
    (["分手", "悲伤", "独处"], "我们分手了，我一个人坐在路边哭了很久。", {"category": "情感"}),
    (["猫", "温暖", "陪伴"], "我的猫趴在我腿上呼噜呼噜地睡觉，好温暖。", {"category": "宠物"}),
    (["量子力学", "纠缠"], "量子纠缠是量子力学中最神秘的现象之一。", {"category": "科学"}),
    (["山中", "自然", "宁静"], "我在山中小屋中醒来，窗外是鸟鸣和阳光。", {"category": "场景"}),
    (
        ["奥运", "金牌", "竞技"],
        "中国在奥运会上取得了多枚金牌，太振奋人心了。",
        {"category": "体育"},
    ),
    (["工作", "压力", "焦虑"], "老板临时安排了很多任务，我感到非常焦虑。", {"category": "职场"}),
    (["抹茶", "奶茶", "饮品"], "今天我尝了新品抹茶奶茶，味道超级好喝！", {"category": "饮食"}),
    (["咖啡", "放松"], "我今天喝了杯咖啡，心情放松了不少。", {"category": "饮食"}),
    (["小猫", "依赖"], "我家小猫越来越黏人了。", {"category": "宠物"}),
    (["奥运", "热血"], "看到奥运比赛我热血沸腾。", {"category": "体育"}),
    (["量子计算", "研究"], "最新的量子计算研究取得了突破。", {"category": "科技"}),
    (["冥想", "山林", "平静"], "在山林中独自冥想，感觉无比平静。", {"category": "心理"}),
    (["考试", "失利", "沮丧"], "这次考试成绩不理想，我有点沮丧。", {"category": "教育"}),
    (["演唱会", "兴奋"], "昨天的演唱会太震撼了，我嗓子都喊哑了！", {"category": "娱乐"}),
    (["回家", "团聚", "温馨"], "春节终于回到家，和爸妈一起吃饭真的很幸福。", {"category": "家庭"}),
    (["旅游", "风景", "放松"], "这次云南之旅让我彻底放松了心情。", {"category": "旅行"}),
    (["高铁", "晚点", "烦躁"], "高铁又晚点了，真的太耽误事了。", {"category": "出行"}),
    (["下雨", "情绪低落"], "连着几天下雨，感觉心情都被压抑了。", {"category": "天气"}),
]

# 为模型类型创建转换后的样本数据
model_samples = [
    (tags, ContentItem(text=text, category=metadata["category"]), metadata)
    for tags, text, metadata in samples
]


def run_tests(vector: TagsVector, samples: list[tuple], vector_type: str) -> None:
    """通用测试运行函数"""
    print(f"\n===== 测试 {vector_type} 类型 =====")

    # 准备批量添加数据
    tags_group = [sample[0] for sample in samples]
    contents = [sample[1] for sample in samples]
    metadatas = [sample[2] for sample in samples]
    ids = [str(i) for i in range(len(samples))]

    # 测试批量添加
    print("批量添加数据...")
    vector.add_tags_group(tags_group, contents, metadatas, ids)

    # 测试单个添加（作为对照）
    """print("\n单个添加数据（对照）...")
    for i, sample in enumerate(samples, start=len(samples)):
        print(f"添加数据 {i}: {sample[1].text if hasattr(sample[1], 'text') else sample[1]}")
        vector.add_tags(*sample, id_=str(i))"""

    # 删除最后一个文档（批量添加的）
    vector.delete_docs([str(len(samples) - 1)])
    print(f"\n已删除文档 {len(samples) - 1}")

    # 查询测试
    queries = [
        "我今天喝了杯拿铁，整个人都放松了",
        "最近压力太大了，工作完全压得我喘不过气",
        "我真的很想念我的小猫，它总是趴在我身边陪着我",
        "高考失败了，我感觉人生都灰暗了",
        "去山里静养几天，听着鸟叫，内心得到平静",
        "看到中国选手拿金牌那一刻我眼泪都下来了",
        "刚从云南回来，那里风景太美了，好想再去一次",
        "听了一场震撼的演唱会，感觉自己被音乐治愈了",
        "这几天一直下雨，心情特别低落",
        "我们终于分手了，我走在街头，不知该去哪",
        "我妈今天做了我最爱吃的红烧肉，超幸福",
        "朋友们为我办了个派对，真的超级感动",
        "昨晚喝了一杯新品奶茶，甜到心坎里",
        "我看到量子计算的新闻，未来真的很奇妙",
        "我家猫最近老往我身上跳，好有安全感",
        "高铁又晚点了，我真的受够了出差",
    ]

    # 查询结果打印
    for query in queries:
        results = vector.similarity_search_with_score(query, extract_high_score=True)
        print(f"\n查询: {query}")
        for doc, score in results:
            print(f"- 内容: {doc.content} | 分数: {score:.4f} | 元数据: {doc.metadata}")


# 运行字符串类型测试
run_tests(str_vector, samples, "字符串")

# 运行模型类型测试
run_tests(model_vector, model_samples, "Pydantic 模型")
