import re
from datasets import load_dataset
from collections import Counter

# 加载数据集
# 请将 'your_dataset_name' 替换为实际的数据集名称
dataset = load_dataset("Alex-xu/secalign-dbg-haiku-javascript-all")

# 用于存储第三方库名称及其引用次数的计数器
library_counter = Counter()

def extract_js_libraries(text):
    # 定义正则表达式模式
    patterns = [
        r"require\(['\"](.*?)['\"]\)",  # require 语句
        r"import (?:.*? from )?['\"](.*?)['\"]",  # import 语句
        r"<script\s+src=['\"](.*?)['\"]"  # script src 语句
    ]
    libraries = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        libraries.extend(matches)
    return libraries


# 遍历数据集中的每个样本
for split in dataset:
    for sample in dataset[split]:
        code = sample['original_instruction'] + sample['original_code']
        libraries = extract_js_libraries(code)
        libraries = list(set(libraries))
        for libirary in libraries:
            library_counter[libirary] += 1
        # # 查找所有匹配的第三方库名称
        # matches = pattern.findall(code)
        # for match in matches:
        #     # 提取库名（忽略模块路径）
        #     library_name = match.split('::')[0]
        #     library_counter[library_name] += 1

# 按引用次数从大到小排序
sorted_libraries = sorted(library_counter.items(), key=lambda item: item[1], reverse=True)

# 打印引用次数超过10的库名
print("引用次数超过10的第三方库：")
for library, count in sorted_libraries:
    if count > 10:
        print(f"{library},",end='')
    else:
        break