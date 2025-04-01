# Requires sentence_transformers>=2.7.0

# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim

# sentences = ['That is a happy person', 'That is a very happy person']

# model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
# embeddings = model.encode(sentences)
# print(embeddings.shape)
# print(cos_sim(embeddings[0], embeddings[1]))

# Requires transformers>=4.36.0

import torch.nn.functional as F
import torch
from transformers import AutoModel, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

model_path = 'Alibaba-NLP/gte-base-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

embedding_model = HuggingFaceEmbeddings(
    model_name='Alibaba-NLP/gte-base-en-v1.5',
    multi_process=False,
    model_kwargs={"device": "cuda:1",
                    'trust_remote_code':True},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**batch_dict)
print(outputs.last_hidden_state.shape)
embeddings = outputs.last_hidden_state[:, 0]



# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings.shape)
print(embeddings[1][:5])

hf_emb = torch.tensor(embedding_model.embed_documents(input_texts))
print(hf_emb.shape)
print(hf_emb[1][:5])
# scores = (embeddings[:1] @ embeddings[1:].T) * 100
# print(scores.tolist())
