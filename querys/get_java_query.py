from datasets import load_dataset
import json
import re

def extract_instruct_and_imports(prompt):
    # 提取 instruct 部分
    start_inst_index = prompt.find("[INST]") + len("[INST]")
    end_inst_index = prompt.find("[/INST]")
    instruct = prompt[start_inst_index:end_inst_index].strip()

    # 提取 code 部分
    start_code_index = end_inst_index + len("[/INST]")
    code = prompt[start_code_index:].strip()

    # 提取 import 语句中的包名
    import_pattern = re.compile(r'import\s+([\w.]+);')
    imports = import_pattern.findall(code)

    return {
        "instruct": instruct,
        "imports": imports
    }

def save_instruct_and_imports_to_json(dataset, output_path):
    result_list = []
    # 遍历数据集中的每个样本
    for example in dataset:
        prompt = example["prompt"]
        # 提取 instruct 和 imports
        result = extract_instruct_and_imports(prompt)
        result_list.append(result)

    # 将结果列表保存为 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)
# 替换为你要下载的数据集名称
dataset_name = "paula-rod-sab/java-exercise-codesc"
# 加载数据集
dataset = load_dataset(dataset_name)["train"]

# 替换为你要保存的 JSON 文件路径
output_path = "java.json"
# 保存 instruct 到 JSON 文件
save_instruct_and_imports_to_json(dataset, output_path)