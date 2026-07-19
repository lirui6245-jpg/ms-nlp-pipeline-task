# 自动生成扩充的模拟数据，用于支持5折交叉验证
import random

positive_templates = [
    "这款产品{}，{}非常{}，强烈推荐。",
    "使用体验{}，{}超出预期，十分满意！",
    "质量{}，做工精良，{}值得信赖。",
    "服务态度{}，响应及时，{}好评。",
    "性价比{}，功能齐全，{}购买。",
    "物流很快，包装完好，{}开心。",
    "设计精美，外观{}，{}喜欢。",
    "性能强劲，运行{}，{}给力。",
    "售后专业，解决问题快，{}放心。",
    "整体优秀，无可挑剔，{}回购。",
]

negative_templates = [
    "这{}太差了，{}有问题，非常失望。",
    "质量堪忧，经常{}，{}后悔。",
    "客服恶劣，态度{}，{}生气。",
    "物流极慢，等了很久，{}烦躁。",
    "虚假宣传，根本{}，{}被骗。",
    "用了就坏，质量{}，{}无语。",
    "价格虚高，不值{}，{}吃亏。",
    "功能缺陷，很多{}，{}崩溃。",
    "包装破损，收到就{}，{}恼火。",
    "体验极差，完全{}，{}差评。",
]

positive_words = ["很好", "不错", "优秀", "出色", "完美", "棒极了", "一流", "极佳", "超级好", "令人满意", "给力", "精致", "顺畅", "专业", "贴心"]
negative_words = ["很差", "糟糕", "垃圾", "极差", "烂透了", "不行", "差劲", "劣质", "坑人", "令人失望", "失望", "破损", "故障", "恶劣", "虚高"]

def generate_corpus(n_per_class=15):
    corpus = []
    
    # 生成正样本
    for i in range(n_per_class):
        template = random.choice(positive_templates)
        # 数一下模板里有几个 {}
        n_slots = template.count("{}")
        words = random.sample(positive_words, n_slots)
        text = template.format(*words)
        # 随机加一些变化
        if random.random() > 0.5:
            text = text.replace("，", "！").replace("。", "！")
        corpus.append((text, 1))
    
    # 生成负样本
    for i in range(n_per_class):
        template = random.choice(negative_templates)
        n_slots = template.count("{}")
        words = random.sample(negative_words, n_slots)
        text = template.format(*words)
        if random.random() > 0.5:
            text = "千万别买！" + text
        corpus.append((text, 0))
    
    return corpus

# 生成30条数据（每类15条），足够支持5折交叉验证
new_corpus = generate_corpus(15)

# 打印成 Python 代码格式
print("RAW_CORPUS = [")
for text, label in new_corpus:
    print(f'    ("{text}", {label}),')
print("]")
