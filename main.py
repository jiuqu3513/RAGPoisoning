# GPU_LIST="0,1,2,3"
GPU_LIST="4,5,6,7"
language_list = ["python","java","javascript",'rust']
language = language_list[2]

# import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_LIST
import pickle
import json 
from tqdm import tqdm
import random
import faiss   
import re
from collections import Counter
import numpy as np
from poisondRAG.models import create_model
from utils.database import load_rag_database, load_query_dataset, load_models, split_documents, create_logger, load_embedding_model
from utils.utils import preprocess_doc,get_sorted_clean_related_docs,preprocess_doc_new
from poisondRAG.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from utils.attacker import Attacker
from poisondRAG.prompts import wrap_prompt
from ragatouille import RAGPretrainedModel
from langchain.docstore.document import Document as LangchainDocument

import ast
import time
from langchain_community.vectorstores import FAISS
from attacker.jailbreak.jailbreaker import Jailbreaker
import hydra
from omegaconf import DictConfig, OmegaConf   

# EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-base-en-v1.5"

@hydra.main(version_base=None, config_path="config", config_name=language)
def main(cfg: DictConfig):
    # torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    gpu_list = [int(item) for item in GPU_LIST.split(",")]
    ngpu = len(gpu_list)
    cfg.gpu_list = cfg.gpu_list[:ngpu]
    resources = [faiss.StandardGpuResources() for i in range(ngpu)]
    setup_seeds(cfg.seed)
    if cfg.rag.model_config_path == False:
        model_config_path = f'model_configs/{cfg.rag.model_name}_config.json'

    language = cfg.rag.language

    original_pkg = cfg.rag.original_pkg
    malicious_pkg = cfg.rag.malicious_pkg

    exp_dir = cfg.rag.exp_dir
    jailbreak_topk = cfg.jb_params.top_k
    eval_model_name = cfg.rag.model_name if cfg.rag.eval_transfer == 1 else cfg.target_llm.llm_params.model_name
    model_save_dir = os.path.join(exp_dir,'logs',cfg.rag.original_pkg,eval_model_name,cfg.target_llm.llm_params.model_name)
    os.makedirs(model_save_dir,exist_ok=True)
    exp_name = f"{malicious_pkg}-eval_{eval_model_name}-target_{cfg.target_llm.llm_params.model_name}-{cfg.rag.attack_method}-jb_first{cfg.rag.jb_first}-usr_r{cfg.rag.use_r}-jb{cfg.rag.use_jb}-rr{cfg.rag.use_rr}-epoch_{cfg.rag.epoch_num}-num_token_{cfg.rag.num_tokens}-beam_width_{cfg.rag.beam_width}-topk{cfg.rag.top_k_tokens}-jbtopk_{jailbreak_topk}"
    poison_database_name = f"{malicious_pkg}-target_{cfg.target_llm.llm_params.model_name}-{cfg.rag.attack_method}-jb_first{cfg.rag.jb_first}-usr_r{cfg.rag.use_r}-jb{cfg.rag.use_jb}-rr{cfg.rag.use_rr}-epoch_{cfg.rag.epoch_num}-num_token_{cfg.rag.num_tokens}-beam_width_{cfg.rag.beam_width}-topk{cfg.rag.top_k_tokens}-jbtopk_{jailbreak_topk}"
    logger_path = os.path.join(model_save_dir,f'{exp_name}.log')
    logger = create_logger(logger_path)
    logger.info(cfg.rag)
    
    # if cfg.rag.original_pkg == 'scrapy':
    #     query_json_path = os.path.join('querys',f'{cfg.rag.original_pkg}.json')
    #     with open(query_json_path, 'r') as f:
    #         ds_json = json.load(f)
    #     target_queries = list(ds_json.values())
    if language == 'java':
        query_json_path = os.path.join('querys',f'{language}',f'{cfg.rag.original_pkg}.json')
        with open(query_json_path, 'r') as f:
            ds_json = json.load(f)
        target_queries = list(ds_json.values())
        # target_queries = [" ".join(sample['instruct']) for _,sample in query_dict.items()]
    elif language == 'python':
        ds_bcb = load_query_dataset('bigcodebench')
        query_dict = {}
        for task_id, task_dataset in ds_bcb.items():
            if original_pkg in task_dataset['libs']:
                query_dict[task_id] = ast.literal_eval(task_dataset['doc_struct'])  # to dict object
        target_queries = [" ".join(doc_struct['description']) for task_id,doc_struct in query_dict.items()]
    elif language == 'rust':
        pattern = re.compile(r'(?:extern crate|use)\s+([\w::]+)(?:\s*;|\s*{)')
        ds_rust = load_query_dataset('neloy_rust_instruction_dataset')
        target_queries = []
        for task_id,sample in enumerate(ds_rust):
            code = sample['output']
            # 查找所有匹配的第三方库名称
            matches = pattern.findall(code)
            for match in matches:
                # 提取库名（忽略模块路径）
                library_name = match.split('::')[0]
                if original_pkg in library_name:
                    target_queries.append(sample['instruction'])
                    break
    elif language == 'javascript':
        ds_js = load_query_dataset('secalign-dbg-haiku-javascript-all')
        query_dict = {}
        for instruct_idx, data_sample in enumerate(ds_js):
            if original_pkg in data_sample['fixed_code'] or original_pkg in data_sample['original_code']:
                # query_dict[task_id] = ast.literal_eval(task_dataset['original_instruction'])  # to dict object
                query_dict[instruct_idx] = data_sample['original_instruction']
        target_queries = ["".join(instruction) for task_id,instruction in query_dict.items()]
        # query_json_path = os.path.join('querys',f'{language}',f'{cfg.rag.original_pkg}.json')
        # with open(query_json_path, 'r') as f:
        #     ds_json = json.load(f)
        # target_queries = list(ds_json.values())
    # assert len(qrels) <= len(results)
    logger.info(f'Total samples:{len(target_queries)}')
    
    if cfg.rag.rerank == 1:
       
        reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    num_docs_final = 5
    # p = args.p 
    num_retrieved_docs = cfg.rag.num_retrieved_docs
    train_retrieved_docs = cfg.rag.train_retrieved_docs

    
    random.shuffle(target_queries) # make sure random split
    # spilt_factor = 0.8 if not language == 'javascript' else 0.75
    spilt_factor = 0.8
    split_index = int(len(target_queries) * spilt_factor)
    train_set = target_queries[:split_index]
    test_set = target_queries[split_index:]

    sorted_clean_related_docs = None
    contriever_name = cfg.rag.contriever_name
    embedding_model=None
    jailbreaker = None
    # Poisoned RAG Database
    poisoned_faiss_index_path = os.path.join(os.getcwd(),exp_dir, "database_faiss",contriever_name,'poison',malicious_pkg,poison_database_name)
    # print(poisoned_faiss_index_path)
    os.makedirs(os.path.join('data',original_pkg),exist_ok=True)
    sorted_clean_related_docs_save_path = os.path.join('data',original_pkg,'sorted_clean_related_docs.pkl')    
    if not os.path.exists(poisoned_faiss_index_path):
        # load clean rag database
        dataset_name_list = ['bcb','ds1000','classeval','humaneval','humanevalx','mbpp','apps']
        rag_database,embedding_model = load_rag_database(dataset_name_list,contriever_name,language=language)
        # if torch.cuda.is_available():
        #     rag_database.index = faiss.index_cpu_to_gpu_multiple_py(resources, rag_database.index)
        sorted_clean_related_docs = get_sorted_clean_related_docs(train_set,rag_database,num_retrieved_docs=train_retrieved_docs,original_pkg=original_pkg)
        if len(sorted_clean_related_docs) == 0:
            logger.info(f"Can not find any related docs about {cfg.rag.original_pkg}")
            raise(f"Can not find any related docs about {cfg.rag.original_pkg}")
        else:
            print('Related Docs: ',len(sorted_clean_related_docs))
        if not os.path.exists(sorted_clean_related_docs_save_path):
            with open(sorted_clean_related_docs_save_path, 'wb') as f:
                pickle.dump(sorted_clean_related_docs, f)

        # Filter out docs with original_pkg
        # Attack
        # if cfg.rag.attack_method not in [None, 'None']:
        # Load retrieval models
        model, tokenizer, get_emb = load_models(cfg.rag.eval_model_code)
        model.eval()
        model.to(device)
        jailbreaker = Jailbreaker(cfg)
        attacker = Attacker(cfg,
                            model=embedding_model,
                            encoder=model,
                            tokenizer=tokenizer,
                            get_emb=get_emb,
                            jailbreaker=jailbreaker,
                            exp_name=exp_name,
                            ) 
        
        logger.info(f"=> Total TBD documents length: {len(sorted_clean_related_docs)}")
        poisoned_docs = {}
        pbar = tqdm(ncols=180,total=len(sorted_clean_related_docs))
        start_time = time.time()
        last_time = start_time
        es_count = 0

        # world_size = torch.cuda.device_count()
        # mp.spawn(attacker.get_attack, args=(world_size,initial_poisoned_doc,query_dist_pair_list,tot_idx,doc,), nprocs=world_size
        cnt = 0
        for doc_idx,(tot_idx, doc_dict) in enumerate(sorted_clean_related_docs):
            # t_time = time.time()
            doc = doc_dict['doc'].page_content
            # if cnt <= 109:
            #     cnt += 1
            #     pbar.update(1)
            #     continue
            if cfg.rag.search_range == 'global':
                initial_poisoned_doc = preprocess_doc_new(doc,original_pkg,malicious_pkg,comment_flag='#' if language == 'python' else '//')
            else:
                initial_poisoned_doc = preprocess_doc(doc,original_pkg,malicious_pkg)
            query_dist_pair_list = doc_dict['query_dist_pair_list']
            final_poisoned_doc,early_stop,final_score,initial_score,r_score,jb_score, rr_score,max_model,max_language,last_time = attacker.get_attack(initial_poisoned_doc,query_dist_pair_list,doc_idx,doc,tot_idx=tot_idx)
            
            logger.info('##################################################')
            logger.info(f'ini_score:{initial_score:.2f} final_score:{final_score:.2f} r_score:{r_score:.2f} jb_score:{jb_score:.2f} rr_score:{rr_score:.2f} model:{max_model} lang:{max_language} query_num:{len(query_dist_pair_list)}\n')
            logger.info(f'doc_idx:{doc_idx}\n Poisoned Doc:\n{final_poisoned_doc}\n')
            if early_stop == 1:
                es_count += 1
            poisoned_docs[doc_idx] = final_poisoned_doc
            # last_time = time.time() - t_time
            # tot_time = time.time() - start_time
            pbar.set_postfix(**{'idx':doc_idx,'qn':len(query_dist_pair_list),'es':es_count,'last': f'{last_time:.2f}s','ini': f'{initial_score:.2f}','final': f'{final_score:.2f}','r': f'{r_score:.2f}','jb': f'{jb_score:.2f}','rr': f'{rr_score:.2f}'})
            pbar.update(1)
            # cnt += 1

        # Add Poisoned Database
        poisoned_RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=doc, metadata={"doc_idx":doc_idx,"ds_idx":doc_idx,"ds":'poisoned'}) for doc_idx,doc in poisoned_docs.items()]
        poisoned_docs_processed = split_documents(
            1024,  # We choose a chunk size adapted to our model
            poisoned_RAW_KNOWLEDGE_BASE,
            tokenizer_name="Alibaba-NLP/gte-base-en-v1.5",
        )    
        rag_database.add_documents(poisoned_docs_processed)
        rag_database.index = faiss.index_gpu_to_cpu(rag_database.index)
        try:
            logger.info('Save poisoned_faiss_index_path Success')
            rag_database.save_local(poisoned_faiss_index_path)  
        except:
            logger.info('Save poisoned_faiss_index_path Failed')
        # rag_database.index = faiss.index_cpu_to_gpu_multiple_py(resources, rag_database.index) # already in cpu
        poisoned_database = rag_database
    else:
        if embedding_model is None:
            embedding_model = load_embedding_model()
        logger.info(f'Using Existing Poisoned Database!')
        poisoned_database = FAISS.load_local(poisoned_faiss_index_path,embedding_model,allow_dangerous_deserialization=True)
        # poisoned_database.index = faiss.index_cpu_to_gpu_multiple_py(resources, poisoned_database.index)

    #########################################################################################
    # Baseline
    baseline_faiss_index_path = os.path.join(os.getcwd(),exp_dir, "database_faiss",contriever_name,'baseline',malicious_pkg)
    if not os.path.exists(baseline_faiss_index_path):
        faiss_index_path = os.path.join(os.getcwd(), "database_faiss",language,contriever_name,'clean')
        baseline_database = FAISS.load_local(faiss_index_path,embedding_model,allow_dangerous_deserialization=True)
        baseline_docs = {}

        if os.path.exists(sorted_clean_related_docs_save_path) and sorted_clean_related_docs is None:
            with open(sorted_clean_related_docs_save_path, 'rb') as f:
                sorted_clean_related_docs = pickle.load(f)

        for doc_idx, (tot_idx, doc_dict) in enumerate(sorted_clean_related_docs):
            doc = doc_dict['doc'].page_content
            if original_pkg not in doc:
                continue
            if cfg.rag.search_range == 'global':
                initial_poisoned_doc = preprocess_doc_new(doc,original_pkg,malicious_pkg,comment_flag='#' if language == 'python' else '//')
            else:
                initial_poisoned_doc = preprocess_doc(doc,original_pkg,malicious_pkg)
            logger.info('********************************************')
            logger.info(f'doc_idx:{doc_idx}\n Baseline Doc:\n{initial_poisoned_doc}\n')
            baseline_docs[doc_idx] = initial_poisoned_doc
        # Add Poisoned Database
        baseline_RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=doc, metadata={"doc_idx":doc_idx,"ds_idx":doc_idx,"ds":'poisoned'}) for doc_idx,doc in baseline_docs.items()]
        baseline_docs_processed = split_documents(
            1024,  # We choose a chunk size adapted to our model
            baseline_RAW_KNOWLEDGE_BASE,
            tokenizer_name="Alibaba-NLP/gte-base-en-v1.5",
        )    
        baseline_database.add_documents(baseline_docs_processed)
        try:
            baseline_database.save_local(baseline_faiss_index_path) 
            logger.info('Save baseline_faiss_index_path Success')
        except:
            logger.info('Save baseline_faiss_index_path Failed')
        
        # baseline_database.index = faiss.index_cpu_to_gpu_multiple_py(resources, baseline_database.index)
    else:
        logger.info(f'Using Existing Baseline Database!')
        baseline_database = FAISS.load_local(baseline_faiss_index_path,embedding_model,allow_dangerous_deserialization=True)
        # baseline_database.index = faiss.index_cpu_to_gpu_multiple_py(resources, baseline_database.index)


    # Evaluation

    if cfg.rag.eval_transfer == 1:
        llm = create_model(model_config_path)
    else:
        if jailbreaker is None:
            jailbreaker = Jailbreaker(cfg)
        llm = jailbreaker.target_llm

    posioned_asr_cnt=0
    posioned_asr_list=[]
    baseline_asr_cnt=0
    baseline_asr_list=[]
    posioned_total_rank = 0
    baseline_total_rank = 0

    pbar = tqdm(ncols=180,total=len(test_set))
    for idx,query in enumerate(test_set):
        logger.info(f'############# Target Question: {idx} #############')
        logger.info(f'Question: {query}\n')
        ############################ Poisoned ##############################
        logger.info("=> Retrieving Poisoned documents...")
        relevant_docs_and_scores = poisoned_database.similarity_search_with_score(query=query, k=num_retrieved_docs)
        page_relevant_docs = []
        poisoned_docs_count = 0
        for doc,score in relevant_docs_and_scores:
            page_relevant_docs.append(doc.page_content)
            if malicious_pkg in doc.page_content:
                poisoned_docs_count += 1
        posioned_total_rank += poisoned_docs_count
        logger.info(f'Poisoned Before Reranking:{poisoned_docs_count}\{num_retrieved_docs}')
        query_prompt = wrap_prompt(query, page_relevant_docs, language=language)
        response = llm.query(query_prompt)
        logger.info(f"-------------Poisoned Output:-------------\n{response}\n\n")
        if clean_str(malicious_pkg) in clean_str(response):
            posioned_asr_cnt += 1 

        ############################ Baseline ##############################
        logger.info("=> Retrieving Baseline documents...")
        relevant_docs_and_scores = baseline_database.similarity_search_with_score(query=query, k=num_retrieved_docs)
        page_relevant_docs = []
        poisoned_docs_count = 0
        for doc,score in relevant_docs_and_scores:
            page_relevant_docs.append(doc.page_content)
            if malicious_pkg in doc.page_content:
                poisoned_docs_count += 1
        baseline_total_rank += poisoned_docs_count
        logger.info(f'Baseline Before Reranking:{poisoned_docs_count}\{num_retrieved_docs}')
        query_prompt = wrap_prompt(query, page_relevant_docs, language=language)
        response = llm.query(query_prompt)
        logger.info(f"-------------Baseline Output:-------------\n{response}\n\n")
        if clean_str(malicious_pkg) in clean_str(response):
            baseline_asr_cnt += 1 
        pbar.update(1)

    average_rank_poison = posioned_total_rank / len(test_set)
    average_rank_base = baseline_total_rank / len(test_set)

    posioned_asr_list.append(posioned_asr_cnt)
    posioned_asr = np.array(posioned_asr_list) / len(test_set)
    posioned_asr_mean = round(np.mean(posioned_asr), 3)
    logger.info(f"Poisoned ASR: {posioned_asr} ASR Mean: {posioned_asr_mean} Poison Rank: {average_rank_poison}\n")

    baseline_asr_list.append(baseline_asr_cnt)
    baseline_asr = np.array(baseline_asr_list) / len(test_set)
    baseline_asr_mean = round(np.mean(baseline_asr), 3)
    logger.info(f"Baseline ASR: {baseline_asr} ASR Mean: {baseline_asr_mean} Baseline rank: {average_rank_base}\n")

    # average_rank_poison,asr_p = calculate_average_rank(logger_path,'Poisoned')
    
    # average_rank_base,asr_base = calculate_average_rank(logger_path,'Baseline')
    

    logger.info(cfg.rag)

if __name__ == '__main__':
    main()