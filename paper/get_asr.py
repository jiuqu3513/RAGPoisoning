import os
import re


def analyze_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        poisoned_asr_match = re.search(r'Poisoned ASR:.*?ASR Mean: (\d+\.\d+)', content)
        baseline_asr_match = re.search(r'Baseline ASR:.*?ASR Mean: (\d+\.\d+)', content)
        poisoned_rank_match = re.search(r'Poison Rank: (\d+\.\d+)', content)
        baseline_rank_match = re.search(r'Baseline rank: (\d+\.\d+)', content)

        poisoned_asr = float(poisoned_asr_match.group(1)) if poisoned_asr_match else None
        baseline_asr = float(baseline_asr_match.group(1)) if baseline_asr_match else None
        poisoned_rank = float(poisoned_rank_match.group(1)) if poisoned_rank_match else None
        baseline_rank = float(baseline_rank_match.group(1)) if baseline_rank_match else None

        return poisoned_asr, baseline_asr, poisoned_rank, baseline_rank


def find_log_files(directory):
    for root, dirs, files in os.walk(directory):
        sub_dir_level = root.replace(directory, '').count(os.sep)
        if sub_dir_level == 3:
            for file in files:
                if file.endswith('.log'):
                    file_path = os.path.join(root, file)
                    poisoned_asr, baseline_asr, poisoned_rank, baseline_rank = analyze_log_file(file_path)
                    if poisoned_asr is not None and baseline_asr is not None and poisoned_rank is not None:
                        sub_dir = os.path.relpath(root, directory)
                        # print(f"子目录: {sub_dir}")
                        print(f"文件名: {file}")
                        # print(f"&{poisoned_asr:.3f}",end=' ') # Poisoned ASR Mean: 
                        print(f"&{baseline_asr:.3f}",end=' ') # Poisoned ASR Mean: 
                        # print(f"&{poisoned_rank:.2f}") # Poison Rank: 
                        print(f"&{baseline_rank:.2f}") # Poison Rank: 
                        # print()


if __name__ == "__main__":
    directory = '/home/kai/poisonCodeGen/packageRAG/exp_python_sear/logs'  # 请将其替换为你实际的目录
    find_log_files(directory)
    