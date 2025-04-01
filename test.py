GPU_LIST="4,5,6,7"
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_LIST
import pickle
from tqdm import tqdm
import random
import faiss
import numpy as np
from poisondRAG.models import create_model
from utils.database import load_rag_database, load_query_dataset, load_models, split_documents, create_logger, get_sorted_clean_related_docs
from poisondRAG.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from utils.attacker import Attacker
from poisondRAG.prompts import wrap_prompt
from ragatouille import RAGPretrainedModel
from langchain.docstore.document import Document as LangchainDocument
import torch
import ast
import time
from langchain_community.vectorstores import FAISS

# from torch.nn.parallel import DistributedDataParallel
# import torch.distributed as dist
# import torch.multiprocessing as mp

# EMBEDDING_MODEL_NAME = "thenlper/gte-small"

def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--gpu_list', type=list, default=[0,1,2,3,4,5,6,7])
    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="hf")
    parser.add_argument('--rag_dataset', type=str, default="ds1000", help='code generation dataset to evaluate')
    parser.add_argument('--eval_dataset', type=str, default="bcb", help='code generation dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir rag_dataset')
    parser.add_argument("--query_results_dir", type=str, default='main')
    parser.add_argument('--rerank', type=int, default=0)
    parser.add_argument('--num_retrieved_docs', type=int, default=10)
    parser.add_argument("--contriever_name", type=str, default='gte_base_en')
    
    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='gpt4o-mini')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--use_truth', type=str, default='True')
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    parser.add_argument('--attack_method', type=str, default='hotflip',choices=['hotflip'])
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=1, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=20, help='one of our parameters, the number of target queries')
    parser.add_argument('-p', type=int, default=0, help='one of our parameters, the number of target queries')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")
    parser.add_argument("--pkg", type=str, default='matplotlib', help="Name of log and result.")
    parser.add_argument('--num_tokens', type=int, default=15, help='length of attack str')
    parser.add_argument('--beam_width', type=int, default=10, help='num of candidates saved for next epoch')
    parser.add_argument('--epoch_num', type=int, default=50, help='num of epochs')
    parser.add_argument('--top_k_tokens', type=int, default=5, help='num of top_k_tokens')

    parser.add_argument('--use_jb', type=int, default=0) 
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    gpu_list = [int(item) for item in GPU_LIST.split(",")]
    ngpu = len(gpu_list)      
    args.gpu_list = args.gpu_list[:ngpu]
    resources = [faiss.StandardGpuResources() for i in range(ngpu)]
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    model_save_dir = os.path.join('logs',args.pkg,args.model_name)
    os.makedirs(model_save_dir,exist_ok=True)
    exp_name = f"{args.pkg}-{args.model_name}-{args.attack_method}-epoch_{args.epoch_num}-num_token_{args.num_tokens}-beam_width_{args.beam_width}-topktok{args.top_k_tokens}"
    logger = create_logger(os.path.join(model_save_dir,f'{exp_name}.log'))
    logger.info(args)
    
    original_pkg = args.pkg

    if args.eval_dataset == 'bcb':
        ds_bcb = load_query_dataset('bigcodebench')
        query_dict = {}
        for task_id, task_dataset in ds_bcb.items():
            if original_pkg in task_dataset['libs']:
                query_dict[task_id] = ast.literal_eval(task_dataset['doc_struct'])  # to dict object

    llm = create_model(args.model_config_path)
    # assert len(qrels) <= len(results)
    logger.info(f'Total samples:{len(query_dict)}')
    
    if args.rerank == 1:
       
        reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    num_docs_final = 5
    p = args.p 
    num_retrieved_docs = args.num_retrieved_docs
    malicious_pkg = f'{original_pkg}_safe'

    target_queries = [" ".join(doc_struct['description']) for task_id,doc_struct in query_dict.items()]
    random.shuffle(target_queries) # make sure random split
    split_index = int(len(target_queries) * 0.8)
    train_set = target_queries[:split_index]
    test_set = target_queries[split_index:]

    sorted_clean_related_docs = None
    contriever_name = args.contriever_name
    # Poisoned RAG Database
    poisoned_faiss_index_path = os.path.join(os.getcwd(), "database_faiss",'gte_base_en','poison',malicious_pkg,exp_name)
    if not os.path.exists(poisoned_faiss_index_path):
        # load clean rag database
        dataset_name_list = ['bcb','ds1000','classeval','humaneval','humanevalx','mbpp','apps']
        rag_database,embedding_model = load_rag_database(dataset_name_list,contriever_name)
        if torch.cuda.is_available():
            rag_database.index = faiss.index_cpu_to_gpu_multiple_py(resources, rag_database.index)
        sorted_clean_related_docs = get_sorted_clean_related_docs(train_set,rag_database,num_retrieved_docs=num_retrieved_docs)
        sorted_clean_related_docs_save_path = os.path.join('data','sorted_clean_related_docs.pkl')
        if not os.path.exists(sorted_clean_related_docs_save_path):
            with open(sorted_clean_related_docs_save_path, 'wb') as f:
                pickle.dump(sorted_clean_related_docs, f)
        # Attack
        if args.attack_method not in [None, 'None']:
            # Load retrieval models
            model, tokenizer, get_emb = load_models(args.eval_model_code)
            model.eval()
            model.to(device)
            attacker = Attacker(args,
                                model=embedding_model,
                                encoder=model,
                                tokenizer=tokenizer,
                                get_emb=get_emb,
                                ) 
        logger.info(f"=> Total TBD documents length: {len(sorted_clean_related_docs)}")
        poisoned_docs = {}
        pbar = tqdm(ncols=150,total=len(sorted_clean_related_docs))
        start_time = time.time()
        last_time = start_time
        es_count = 0

        # world_size = torch.cuda.device_count()
        # mp.spawn(attacker.get_attack, args=(world_size,initial_poisoned_doc,query_dist_pair_list,tot_idx,doc,), nprocs=world_size)
        cnt = 0
        for tot_idx, doc_dict in sorted_clean_related_docs:
            t_time = time.time()
            doc = doc_dict['doc'].page_content
            if original_pkg not in doc:
                continue
            # if cnt <= 31:
            #     cnt += 1
            #     pbar.update(1)
            #     continue
            initial_poisoned_doc = attacker.preprocess_doc(doc,original_pkg)
            query_dist_pair_list = doc_dict['query_dist_pair_list']
            poisoned_doc,early_stop = attacker.get_attack(initial_poisoned_doc,query_dist_pair_list,tot_idx,doc)
            # logger.info('##################################################')
            # logger.info(f'tot_idx:{tot_idx}\nPoisoned Doc:\n{poisoned_doc}\n')
            if early_stop == 1:
                es_count += 1
            poisoned_docs[tot_idx] = poisoned_doc
            last_time = time.time() - t_time
            tot_time = time.time() - start_time
            pbar.set_postfix(**{'doc_idx':tot_idx,'q_num':len(query_dist_pair_list),'es_num':es_count,'tot_time': f'{tot_time:.2f}s','last': f'{last_time:.2f}s'})
            pbar.update(1)
            # cnt += 1

        # Add Poisoned Database
        poisoned_RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=doc, metadata={"tot_idx":tot_idx,"ds_idx":tot_idx,"ds":'poisoned'}) for tot_idx,doc in poisoned_docs.items()]
        poisoned_docs_processed = split_documents(
            1024,  # We choose a chunk size adapted to our model
            poisoned_RAW_KNOWLEDGE_BASE,
            tokenizer_name="Alibaba-NLP/gte-base-en-v1.5",
        )    
        rag_database.add_documents(poisoned_docs_processed)
        rag_database.index = faiss.index_gpu_to_cpu(rag_database.index)
        try:
            rag_database.save_local(poisoned_faiss_index_path)  
        except:
            print("save poisoned_faiss_index_path error")
        rag_database.index = faiss.index_cpu_to_gpu_multiple_py(resources, rag_database.index)
        poisoned_database = rag_database
    else:
        poisoned_database = FAISS.load_local(poisoned_faiss_index_path,embedding_model,allow_dangerous_deserialization=True)
        poisoned_database.index = faiss.index_cpu_to_gpu_multiple_py(resources, poisoned_database.index)

    #########################################################################################
    # Baseline
    baseline_faiss_index_path = os.path.join(os.getcwd(), "database_faiss",'gte_base_en','baseline',malicious_pkg)
    if not os.path.exists(baseline_faiss_index_path):
        faiss_index_path = os.path.join(os.getcwd(), "database_faiss",'gte_base_all_bcb','clean')
        baseline_database = FAISS.load_local(faiss_index_path,embedding_model,allow_dangerous_deserialization=True)
        baseline_docs = {}

        if os.path.exists(sorted_clean_related_docs_save_path) and sorted_clean_related_docs is None:
            with open(sorted_clean_related_docs_save_path, 'rb') as f:
                sorted_clean_related_docs = pickle.load(f)

        for tot_idx, doc_dict in sorted_clean_related_docs:
            doc = doc_dict['doc'].page_content
            if original_pkg not in doc:
                continue
            initial_poisoned_doc = attacker.preprocess_doc(doc,original_pkg)
            logger.info('********************************************')
            logger.info(f'Tot_idx:{tot_idx}\n Baseline Doc:\n{initial_poisoned_doc}\n')
            baseline_docs[tot_idx] = initial_poisoned_doc
        # Add Poisoned Database
        baseline_RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=doc, metadata={"tot_idx":tot_idx,"ds_idx":tot_idx,"ds":'poisoned'}) for tot_idx,doc in baseline_docs.items()]
        baseline_docs_processed = split_documents(
            1024,  # We choose a chunk size adapted to our model
            baseline_RAW_KNOWLEDGE_BASE,
            tokenizer_name="Alibaba-NLP/gte-base-en-v1.5",
        )    
        try:
            baseline_database.save_local(baseline_faiss_index_path) 
            # baseline_database = 
            # faiss.write_index(baseline_database, os.path.join(baseline_faiss_index_path,"index.faiss")) 
        except:
            print("save baseline_faiss_index_path error")
        baseline_database.add_documents(baseline_docs_processed)
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(resources, baseline_database.index)
        baseline_database.index = gpu_index
    else:
        baseline_database = FAISS.load_local(baseline_faiss_index_path,embedding_model,allow_dangerous_deserialization=True)
        baseline_database.index = faiss.index_cpu_to_gpu_multiple_py(resources, baseline_database.index)


    # Evaluation
    posioned_asr_cnt=0
    posioned_asr_list=[]
    baseline_asr_cnt=0
    baseline_asr_list=[]
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
        logger.info(f'Poisoned Before Reranking:{poisoned_docs_count}\{num_retrieved_docs}')
        query_prompt = wrap_prompt(query, page_relevant_docs, 4)
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
        logger.info(f'Baseline Before Reranking:{poisoned_docs_count}\{num_retrieved_docs}')
        query_prompt = wrap_prompt(query, page_relevant_docs, 4)
        response = llm.query(query_prompt)
        logger.info(f"-------------Baseline Output:-------------\n{response}\n\n")
        if clean_str(malicious_pkg) in clean_str(response):
            baseline_asr_cnt += 1 

    posioned_asr_list.append(posioned_asr_cnt)
    posioned_asr = np.array(posioned_asr_list) / len(test_set)
    posioned_asr_mean = round(np.mean(posioned_asr), 2)
    logger.info(f"Poisoned ASR: {posioned_asr}\nASR Mean: {posioned_asr_mean}\n")

    baseline_asr_list.append(baseline_asr_cnt)
    baseline_asr = np.array(baseline_asr_list) / len(test_set)
    baseline_asr_mean = round(np.mean(baseline_asr), 2)
    logger.info(f"Baseline ASR: {baseline_asr}\nASR Mean: {baseline_asr_mean}\n")
    

if __name__ == '__main__':
    main()