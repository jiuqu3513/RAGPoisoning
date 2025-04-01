from sentence_transformers import SentenceTransformer
import torch
# import random
# from tqdm import tqdm
from poisondRAG.utils import load_json
import json
import os
from utils.attack.hotflip import HotFlip
from utils.attack.sear import SEAR
# from utils.attack.cold import COLD
from scipy.spatial.distance import cosine

class Attacker():
    def __init__(self, cfg, **kwargs) -> None:
        # assert args.attack_method in ['default', 'whitebox']
        self.exp_name = kwargs.get('exp_name', None)
        self.model = kwargs.get('model', None)
        self.encoder = kwargs.get('encoder', None)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.get_emb = kwargs.get('get_emb', None)
        self.cfg = cfg
        jailbreaker = kwargs.get('jailbreaker', None)
        
        if cfg.rag.attack_method == 'sear':
            self.optimizer = SEAR(encoder=self.encoder,tokenizer=self.tokenizer,hf_model=self.model,jailbreaker=jailbreaker,cfg=cfg)
        else:
            self.optimizer = HotFlip(encoder=self.encoder,tokenizer=self.tokenizer,hf_model=self.model,jailbreaker=jailbreaker,cfg=cfg)
        self.optimizer_name = self.optimizer.__class__.__name__

    def compute_hf_avg_sim(self,embeddings):
        total_similarity = 0
        num_pairs = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                total_similarity += similarity
                num_pairs += 1

        if num_pairs > 0:
            average_similarity = total_similarity / num_pairs
            print("document cos sim", average_similarity)
    
    def get_attack(self, initial_poisoned_doc, query_dist_pair_list,doc_idx,clean_doc,tot_idx=0):
        '''
        This function returns adv_text_groups, which contains adv_texts for M queries
        For each query, if adv_per_query>1, we use different generated adv_texts or copies of the same adv_text
        '''
        log_dir = os.path.join(self.cfg.rag.exp_dir,'resume',self.cfg.rag.original_pkg,self.optimizer_name,self.cfg.rag.contriever_name,self.cfg.target_llm.llm_params.model_name)
        os.makedirs(log_dir,exist_ok=True)

        
        log_path = os.path.join(log_dir, f'{self.exp_name}.log')
        result_json = {}

        # 尝试读取现有日志文件
        try:
            with open(log_path, 'r') as f:
                result_json = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # 文件不存在或内容无效时初始化空字典
            result_json = {}

        # 获取当前实验的tot_idx
        # tot_idx = result_dict["tot_idx"]
        str_doc_idx = str(doc_idx)
        if str_doc_idx in result_json:
            # 当tot_idx存在时，直接使用已保存的结果
            print("Finding existing result!!!")
            result_dict = result_json[str_doc_idx]
            attack_result = result_dict[self.optimizer_name]
            final_poisoned_doc = attack_result['final_poisoned_doc']
            r_str,jb_str,rr_str = attack_result['r_str'],attack_result['jb_str'],attack_result['rr_str']
            best_score,initial_score,r_score,jb_score, rr_score = attack_result['best_score'],attack_result['initial_score'],attack_result['r_score'],attack_result['jb_score'],attack_result['rr_score']
            early_stop,max_model, max_language = int(attack_result['early_stop']),attack_result['max_model'],attack_result['max_language']
            time = float(attack_result['time'])
        else:
            # 当tot_idx不存在时，获取结果并保存
            # result_dict = self.get_result(tot_idx)  # 如果需要动态获取
            queries = [pair[0] for pair in query_dist_pair_list]
            result_dict = {'doc_idx':doc_idx,
                           'tot_idx':tot_idx,
                        'query_num':len(queries)} 
            result_dict.setdefault(self.optimizer_name, {})
            final_poisoned_doc,r_str,jb_str,rr_str,best_score,initial_score,r_score,jb_score, rr_score,early_stop,max_model, max_language,time = self.optimizer.attack(query_list=queries,initial_poisoned_doc=initial_poisoned_doc,clean_doc=clean_doc)
            result_dict[self.optimizer_name] = {'r_str':r_str, 'jb_str':jb_str, 'rr_str':rr_str, 
                                                'initial_score':initial_score if initial_score == -1 else initial_score.tolist(), 
                                                'best_score':best_score.tolist(),
                                                'r_score':r_score if r_score == -1 else r_score.tolist(), 
                                                'jb_score':jb_score if jb_score == -1 else jb_score.tolist(),
                                                'rr_score':rr_score if rr_score == -1 else rr_score.tolist(),
                                                'max_model':max_model, 
                                                'max_language':max_language,
                                                'early_stop':early_stop,
                                                'time':time,
                                                'final_poisoned_doc':final_poisoned_doc}
            result_json[doc_idx] = result_dict
            with open(log_path, 'w') as f:
                json.dump(result_json, f, indent=2)

        # try:
        #     with open(os.path.join(log_dir,f'{self.exp_name}.log'), 'rw') as f:
        #         result_json = json.load(f)
        #         result_json[result_dict["tot_idx"]] = result_dict
        #         json.dump(result_json, f, indent=2)
        # except:
        #     result_json = {result_dict["tot_idx"]: result_dict}
        #     with open(os.path.join(log_dir,f'{self.exp_name}.log'), 'w') as f:
        #         json.dump(result_json, f, indent=2)

        return final_poisoned_doc,early_stop,best_score,initial_score,r_score,jb_score, rr_score,max_model, max_language
