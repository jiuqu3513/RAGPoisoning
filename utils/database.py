import sys, os
from .contriever_src.contriever import Contriever
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json
from datasets import load_dataset
import numpy as np
from collections import defaultdict
import random
import torch
import re
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from bigcodebench.data import get_bigcodebench
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document as LangchainDocument
from typing import Optional, List, Tuple

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

import logging

# EMBEDDING_MODEL_NAME = "thenlper/gte-small"
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-base-en-v1.5"
# EMBEDDING_MODEL_NAME = "dunzhang/stella_en_400M_v5"


MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
# CODE_SEPARATORS = [
#     "\n\n\n\n",
# ]
model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def load_models(model_code):
    # assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb

        # tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
        # model = AutoModel.from_pretrained("thenlper/gte-small")
        # c_model = model
        # get_emb = contriever_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    elif 'hf' in model_code:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME,trust_remote_code=True)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        model.resize_token_embeddings(len(tokenizer))  # 30522->30528 with 6 additional tokens
        get_emb = contriever_get_emb
    else:
        raise NotImplementedError
    
    return model, tokenizer, get_emb

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique
    

def load_query_dataset(dataset_name,subset='full'):
    # assert dataset_name in ['bigcodebench']

    if dataset_name == 'bigcodebench':
        query_ds = get_bigcodebench(subset=subset)
    elif dataset_name == 'secalign-dbg-haiku-javascript-all':
        query_ds = load_dataset("Alex-xu/secalign-dbg-haiku-javascript-all", split="train")
    elif dataset_name == 'neloy_rust_instruction_dataset':
        query_ds = load_dataset("Neloy262/rust_instruction_dataset",split='train')
    return query_ds

def load_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=False,
        model_kwargs={"device": "cuda",
                      'trust_remote_code':True},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )
    return embedding_model
    
def load_rag_database(dataset_name_list,contriever_name,language='python'):
    # embedding model 
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=False,
        model_kwargs={"device": "cuda",
                      'trust_remote_code':True},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    if language == 'python':
        faiss_index_path = os.path.join(os.getcwd(), "database_faiss",'python',contriever_name,'clean')
        if not os.path.exists(faiss_index_path):
            RAW_KNOWLEDGE_BASE = []
            doc_idx = 0
            for dataset_name in dataset_name_list:
                if dataset_name == 'bcb':
                    # Python 3.0+
                    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['complete_prompt'] + '\n' +task['canonical_solution'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)                
                elif dataset_name == 'ds1000':
                    # Python
                    ds = load_dataset("xlangai/DS-1000",split='test')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['prompt'] + '\n' +task['code_context'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'classeval':
                    # Python
                    ds = load_dataset("FudanSELab/ClassEval",split='test')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['skeleton'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'humaneval':
                    # Python
                    ds = load_dataset("openai/openai_humaneval",split='test')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['prompt'] + '\n' +task['canonical_solution'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'humanevalx':
                    # Python, C++, Java, JavaScript, and Go
                    ds = load_dataset("THUDM/humaneval-x",'python')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['prompt'] + '\n' +task['canonical_solution'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds['test']))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'mbpp':
                    # Python
                    ds = load_dataset("google-research-datasets/mbpp")
                    for split_name in ['train','test','validation','prompt']:
                        RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['text'] + '\n' +task['code'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx,task in tqdm(enumerate(ds[split_name]))])
                        doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'apps':
                    # Python
                    ds = load_dataset("codeparrot/apps", split="all",trust_remote_code=True)
                    for split_name in ['train','test']:
                        RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['question'] + '\n' +task['solutions'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])             
                        doc_idx = len(RAW_KNOWLEDGE_BASE)
            docs_processed = split_documents(
                1024,  # We choose a chunk size adapted to our model
                RAW_KNOWLEDGE_BASE,
                tokenizer_name=EMBEDDING_MODEL_NAME,
            )
            KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
                docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
            KNOWLEDGE_VECTOR_DATABASE.save_local(faiss_index_path)  
        else:
            print("Using Existing FAISS Database!")
            KNOWLEDGE_VECTOR_DATABASE=FAISS.load_local(faiss_index_path,embedding_model,allow_dangerous_deserialization=True)
    elif language == 'javascript':
        dataset_name_list_js = ['buzz_sources_042_javascript','secalign','genesys_api_javascript_alpaca','Evol-Instruct-JS-Code-500-v1','Evol-Instruct-JS-1k'] # 'codex_js'
        faiss_index_path = os.path.join(os.getcwd(), "database_faiss",'javascript',contriever_name,'clean')
        if not os.path.exists(faiss_index_path):
            RAW_KNOWLEDGE_BASE = []
            doc_idx = 0
            for dataset_name in dataset_name_list_js:
                # if dataset_name == 'js':
                #     # Python
                #     ds = load_dataset("hchautran/javascript-small",split='train')
                #     RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['content'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                #     doc_idx = len(RAW_KNOWLEDGE_BASE)
                if dataset_name == 'buzz_sources_042_javascript':
                    # instruction + output
                    ds = load_dataset("supergoose/buzz_sources_042_javascript", split="train")
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=str(task['conversations'][0]), metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'secalign':
                    ds = load_dataset("Alex-xu/secalign-dbg-haiku-javascript-all",'default')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['original_instruction'] + '\n' +task['original_code']+ '\n' +task['fixed_code'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds['train']))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'genesys_api_javascript_alpaca':
                    # instruction + output
                    ds = load_dataset("cmaeti/genesys_api_javascript_alpaca",'default')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['instruction'] + '\n' +task['output'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds['train']))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'Evol-Instruct-JS-Code-500-v1':
                    # instruction + output
                    ds = load_dataset("pyto-p/Evol-Instruct-JS-Code-500-v1",'default')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['instruction'] + '\n' +task['output'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds['train']))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'Evol-Instruct-JS-1k':
                    # instruction + output
                    ds = load_dataset("harryng4869/Evol-Instruct-JS-1k",'default')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['instruction'] + '\n' +task['output'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds['train']))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)

            docs_processed = split_documents(
                1024,  # We choose a chunk size adapted to our model
                RAW_KNOWLEDGE_BASE,
                tokenizer_name=EMBEDDING_MODEL_NAME,
            )
            KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
                docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
            KNOWLEDGE_VECTOR_DATABASE.save_local(faiss_index_path)  
        else:
            print("Using Existing FAISS Database!")
            KNOWLEDGE_VECTOR_DATABASE=FAISS.load_local(faiss_index_path,embedding_model,allow_dangerous_deserialization=True)
    elif language == 'java':
        dataset_name_list_java = ['stack_java','java-exercise'] # 'codex_js'
        faiss_index_path = os.path.join(os.getcwd(), "database_faiss",'java',contriever_name,'clean')
        if not os.path.exists(faiss_index_path):
            RAW_KNOWLEDGE_BASE = []
            doc_idx = 0
            for dataset_name in dataset_name_list_java:
                if dataset_name == 'stack_java':
                    # 
                    ds = load_dataset("ammarnasr/the-stack-java-clean", split="train")
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['content'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)                
                elif dataset_name == 'java-exercise':
                    '''
                    <s>[INST] Write a Java program that checks whether an array is negative dominant or not. If the array is negative dominant return true otherwise false.   Example:
                        Original array of numbers:
                        [1, -2, -5, -4, 3, -6]
                        Check Negative Dominance in the said array!true [/INST]
                        // Import necessary Java classes.
                        import java.util.Scanner;
                        import java.util.Arrays;
                        </s>
                    '''
                    ds = load_dataset("paula-rod-sab/java-exercise-codesc",split='train')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['prompt'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
            docs_processed = split_documents(
                1024,  # We choose a chunk size adapted to our model
                RAW_KNOWLEDGE_BASE,
                tokenizer_name=EMBEDDING_MODEL_NAME,
            )
            KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
                docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
            KNOWLEDGE_VECTOR_DATABASE.save_local(faiss_index_path)  
        else:
            print("Using Existing FAISS Database!")
            KNOWLEDGE_VECTOR_DATABASE=FAISS.load_local(faiss_index_path,embedding_model,allow_dangerous_deserialization=True)
    elif language == 'rust':
        dataset_name_list_rust = ['humaneval-rust','RustBioGPT','rust_instruction_dataset','neloy_rust_instruction_dataset'] # 'codex_js'
        faiss_index_path = os.path.join(os.getcwd(), "database_faiss",'rust',contriever_name,'clean')
        if not os.path.exists(faiss_index_path):
            RAW_KNOWLEDGE_BASE = []
            doc_idx = 0
            for dataset_name in dataset_name_list_rust:
                if dataset_name == 'humaneval-rust':
                    ds = load_dataset("diversoailab/humaneval-rust", split="train")
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['prompt'] + '\n' + task['declaration'] + '\n' +task['cannonical_solution'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)                
                elif dataset_name == 'RustBioGPT':
                    ds = load_dataset("jelber2/RustBioGPT",split='train')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['content'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'rust_instruction_dataset':
                    ds = load_dataset("ysr/rust_instruction_dataset",split='train')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['instruction'] + '\n' + task['code'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
                elif dataset_name == 'neloy_rust_instruction_dataset':
                    ds = load_dataset("Neloy262/rust_instruction_dataset",split='train')
                    RAW_KNOWLEDGE_BASE.extend([LangchainDocument(page_content=task['instruction'] + '\n' + task['output'], metadata={"tot_idx":doc_idx+ds_idx,"ds_idx":ds_idx,"ds":dataset_name}) for ds_idx, task in tqdm(enumerate(ds))])
                    doc_idx = len(RAW_KNOWLEDGE_BASE)
            docs_processed = split_documents(
                1024,  # We choose a chunk size adapted to our model
                RAW_KNOWLEDGE_BASE,
                tokenizer_name=EMBEDDING_MODEL_NAME,
            )
            KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
                docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
            KNOWLEDGE_VECTOR_DATABASE.save_local(faiss_index_path)  
        else:
            print("Using Existing FAISS Database!")
            KNOWLEDGE_VECTOR_DATABASE=FAISS.load_local(faiss_index_path,embedding_model,allow_dangerous_deserialization=True)
    return KNOWLEDGE_VECTOR_DATABASE,embedding_model

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'results/query_results/{dir}'):
        os.makedirs(f'results/query_results/{dir}', exist_ok=True)
    with open(os.path.join(f'results/query_results/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_results(file_name):
    with open(os.path.join('results', file_name)) as file:
        results = json.load(file)
    return results

def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder,indent=4)
    dict_from_str = json.loads(json_dict)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f,indent=4)

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()

def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.
    
    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.
    
    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    
    return f1_scores

def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(log_path,mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

