def calculate_average_rank(log_file_path, name):
    total_count = 0
    total_rank = 0
    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if f'{name} Before Reranking' in line:
                total_count += 1
                rank = int(line.split('Reranking:')[1].split('\\')[0])
                total_rank += rank

            if f'{name} ASR:' in line:
                        parts = line.split('[')
                        if len(parts)>1:
                            asr = parts[1].split(']')[0]
                            # print(asr)
                            
    if total_count == 0:
        return 0,asr
    else:
        return total_rank / total_count,asr

if __name__ == "__main__":
    log_file_path = '/home/kai/poisonCodeGen/packageRAG/exp_jb_multi_suggestion/logs/matplotlib/gpt4o-mini/llama-3.2-3b-instruct/matplotlib_safe-gpt4o-mini-llama-3.2-3b-instruct-hotflip-jb0-epoch_50-num_token_15-beam_width_10-topk5-jbtopk_16.log'  # 替换为你的日志文件路径
    average_rank_base,asr_base = calculate_average_rank(log_file_path,'Baseline')
    average_rank_poison,asr_p = calculate_average_rank(log_file_path,'Poisoned')

    print(f"Baseline rank: {average_rank_base}",'asr:',asr_base)
    print(f"Poison rank: {average_rank_poison}",'asr:',asr_p)