import sys, os
from .contriever_src.contriever import Contriever
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json
import datasets
import numpy as np
from collections import defaultdict
import random
import torch
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from bigcodebench.data import get_bigcodebench
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document as LangchainDocument
from typing import Optional, List, Tuple

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import logging


EMBEDDING_MODEL_NAME = "thenlper/gte-small"
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
CODE_SEPARATORS = [
    "\n\n\n\n",
]
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
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    
    return model, c_model, tokenizer, get_emb

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
        separators=CODE_SEPARATORS,
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
    
def load_beir_datasets(dataset_name, split):
    assert dataset_name in ['nq', 'msmarco', 'hotpotqa']
    if dataset_name == 'msmarco': split = 'train'
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset_name)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        split = 'train'
    corpus, queries, qrels = data.load(split=split)    

    return corpus, queries, qrels


# def load_ds1000_datasets():
#     from datasets import load_dataset
#     ds = load_dataset("xlangai/DS-1000")
#     return ds

def load_bigcodebench_datasets(dataset_name, dataset_only=False,split='train',subset='full'):
    assert dataset_name in ['bigcodebench','ds1000']
    faiss_index_path = os.path.join(os.getcwd(), "database_faiss",dataset_name,'clean')
    if dataset_name == 'bigcodebench':
        ds = get_bigcodebench(subset=subset)
    else:
        from datasets import load_dataset
        ds = load_dataset("xlangai/DS-1000",split='test')
    if dataset_only:
        return ds

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )
    
    if not os.path.exists(faiss_index_path):

        if dataset_name == 'bigcodebench':
            RAW_KNOWLEDGE_BASE = [
                LangchainDocument(page_content=task["test"], metadata={"task_id":task_id, "libs": task["libs"]}) for task_id,task in tqdm(ds.items())
            ]
        else:
            RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=task['prompt'] + '\n' +task['code_context'], metadata={"metadata":task['metadata']}) for task in tqdm(ds)]
        # task["doc_struct"]['descriptions'] 

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
        KNOWLEDGE_VECTOR_DATABASE=FAISS.load_local(faiss_index_path,embedding_model,allow_dangerous_deserialization=True)

    return KNOWLEDGE_VECTOR_DATABASE,ds,embedding_model

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
