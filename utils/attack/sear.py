import torch
import os
import time
from torch.nn.functional import normalize
import re
from tqdm import tqdm
from scipy.spatial.distance import cosine
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import SUGGESTIONS_DICT
from sklearn.metrics.pairwise import cosine_similarity

class FindNoise(torch.nn.Module):
    def __init__(self, model,noise):
        super(FindNoise,self).__init__()
        self.model = model
        self.noise = noise
        self.noise.requires_grad = True
    def forward(self,input,start,end):
        # input = input + self.noise
        # input[:,start:end,:] += self.noise
        C = torch.cat((input[:,:start,:], input[:,start:end,:]+self.noise,input[:,end:,:]), dim=1) 
        # C = torch.clamp(C, -1.0, 1.0)
        outputs = self.model(inputs_embeds=C) 
        return outputs 
    def get_adv(self,input,start,end):
        C = torch.cat((input[:,:start,:], input[:,start:end,:]+self.noise,input[:,end:,:]), dim=1) 
        # C = torch.clamp(C, -1.0, 1.0)
        # input[:,start:end,:] += self.noise
        return C
    
# Function to disable gradient computation
def set_no_grad(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
class SEAR:
    def __init__(self,tokenizer=None,encoder=None,hf_model=None,jailbreaker=None,cfg=None):
        # Initialize the encoder and tokenizer
        self.encoder_tokenizer = tokenizer
        self.encoder = encoder
        self.device_ids = cfg.gpu_list
        self.vocab_size = self.encoder_tokenizer.vocab_size
        self.hf_model = hf_model
        self.encoder_word_embedding = self.encoder.get_input_embeddings().weight.detach()
        # self.encoder_word_embedding = self.encoder_word_embedding / torch.norm(self.encoder_word_embedding, dim=1, keepdim=True)
        # self.encoder_word_embedding = self.encoder._first_module().word_embeddings.weight
        self.device = self.encoder.device
        self.encoder = DataParallel(self.encoder, device_ids=self.device_ids)

        # self.hf_model = DataParallel(self.hf_model, device_ids=self.device_ids)
        # self.hf_model.to(self.device_ids[0])
        set_no_grad(self.encoder)

        self.initial_poisoned_doc_input_ids = None
        self.initial_poisoned_doc_tokenized_text = None
        self.initial_poisoned_doc_embs = None
        self.offset_mapping = None
        self.query_list_embs = None
        self.clean_score = None

        # hyperparameters
        self.num_tokens = cfg.rag.num_tokens
        self.beam_width = cfg.rag.beam_width
        self.epoch_num = cfg.rag.epoch_num
        self.rr_epoch_num = cfg.rag.rr_epoch_num
        self.top_k_tokens = cfg.rag.top_k_tokens
        self.initial_index = 0
        self.patience = 5  
        # OOM
        self.max_token_length_4 = 700
        self.max_token_length_8 = 2100
        self.max_token_length_long = 1200
        self.max_batch_size = 750
        self.max_total_length = 750000

        self.use_jb = cfg.rag.use_jb
        self.use_rr = cfg.rag.use_rr
        self.use_r = cfg.rag.use_r
        self.jb_first = cfg.rag.jb_first
        self.head_insert = cfg.rag.head_insert
        self.search_range = cfg.rag.search_range
        self.original_pkg = cfg.rag.original_pkg
        self.malicious_pkg = cfg.rag.malicious_pkg
        self.jailbreaker = jailbreaker

        self.noise_model = None
    def compute_query_embs(self, query_list):
        self.query_list_embs = torch.tensor(self.hf_model.embed_documents(query_list),device=self.device) # already normarlize 

    # Function to compute document embeddings
    def compute_doc_embs(self,initial_poisoned_doc,clean_doc,model='baseline',language='English'):
        initial_poisoned_doc_token = self.encoder_tokenizer(initial_poisoned_doc, padding=False,truncation=True,return_tensors='pt', return_offsets_mapping=True).to(self.device)
        self.initial_poisoned_doc_sequence = initial_poisoned_doc_token['input_ids'][0]
        self.offset_mapping = initial_poisoned_doc_token["offset_mapping"][0]
        self.initial_poisoned_doc_tokenized_text = self.encoder_tokenizer.tokenize(initial_poisoned_doc,padding=False,truncation=True)
        if self.search_range == 'global':
            flag_text = SUGGESTIONS_DICT[model][language].replace('malicious_pkg',self.malicious_pkg).replace('original_pkg',self.original_pkg)
            # end_position_in_text = initial_poisoned_doc.find(flag_text) + len(flag_text)
            match = re.search(flag_text, initial_poisoned_doc)
            # 映射字符位置到token位置\
            target_token_pos = -1
            # print(initial_poisoned_doc)
            if not match is None:
                location = match.end()
            else:
                location = initial_poisoned_doc.rfind(f'use {self.original_pkg} anymore') + len(f'use {self.original_pkg} anymore')
            for i, (token_start, token_end) in enumerate(self.offset_mapping):
                if token_start <= location - 1 < token_end:
                    target_token_pos = i
                    break
            self.initial_index = target_token_pos + 1
  
        else:
            flag = '000'
            try:
                self.initial_index = self.initial_poisoned_doc_tokenized_text.rfind(flag) + len(flag) 
            except:
                index_list1 = [index for index, token in enumerate(self.initial_poisoned_doc_tokenized_text) if flag in token]
                if not len(index_list1) == 0:
                    self.initial_index = index_list1[0] + len(flag) -1
                else:
                    self.initial_index = 0 # just set as the start index        
        self.clean_score = torch.matmul(torch.tensor(self.hf_model.embed_query(clean_doc),device=self.device), self.query_list_embs.t())

    
    def insert_into_sequence_global(self,inserted_tokens,pos):
        # compute original document embeddings, not query
        # insert_index = len(self.encoder_tokenizer.convert_tokens_to_ids(self.initial_poisoned_doc_tokenized_text[:pos]))
        # 合并分词结果
        combined_tokens = torch.cat([self.initial_poisoned_doc_sequence[:pos], inserted_tokens,self.initial_poisoned_doc_sequence[pos:]])
        positions = list(range(pos,pos+len(inserted_tokens)))
        # combined_tokens = original_text_tokens[:insert_index] + inserted_tokens + original_text_tokens[insert_index:]
        return combined_tokens,positions
        
    def insert_into_sequence(self,inserted_tokens):
        # compute original document embeddings, not query
        original_text_tokens = self.initial_poisoned_doc_sequence
        insert_index = len(self.encoder_tokenizer.convert_tokens_to_ids(self.initial_poisoned_doc_tokenized_text[:self.initial_index]))
        if self.head_insert == 1:
            combined_tokens = torch.cat([inserted_tokens,original_text_tokens])
        else: 
            # 合并分词结果
            combined_tokens = torch.cat([original_text_tokens[:insert_index], inserted_tokens,original_text_tokens[insert_index+1:]]) # TBD:del '000'
        positions = list(range(insert_index,insert_index+len(inserted_tokens)))    
            # combined_tokens = original_text_tokens[:insert_index] + inserted_tokens + original_text_tokens[insert_index:]
        return combined_tokens,positions
    
    # Function to compute the score (cosine similarity) of a sequence
    def compute_sequence_score(self, sequence,loc=None):
        # querys_embs_list: normarlized, List of embeddings, one for each text.
        # sequence = torch.tensor(sequence, device=device)
        if self.search_range == 'global':
            sequence,_ = self.insert_into_sequence_global(sequence,loc) # print(len(sequence)) # 498
        else:
            sequence,_ = self.insert_into_sequence(sequence) # print(len(sequence)) # 498
        input_ids = sequence.unsqueeze(0)
        with torch.no_grad():
            outputs = self.encoder(input_ids)
        query_embeds = outputs.last_hidden_state[:,0]
        score = torch.matmul(normalize(query_embeds, p=2, dim=1), self.query_list_embs.t())
        mean_score = score.mean().detach().cpu().numpy()
        return mean_score

    def split_encode(self,sequence_batch,split_size=4):
        # split_size = sequence_batch.size(0) // (split)
        
        split_batches = torch.split(sequence_batch, split_size)
        outputs_list = []
        for split_batch in split_batches: 
            # split_batch = split_batch.to(self.device_ids[0])
            with torch.no_grad():
                outputs = self.encoder(split_batch)
            # torch.cuda.empty_cache()
            outputs_list.append(outputs.last_hidden_state[:,0])
        query_embeds_batch = torch.cat(outputs_list, dim=0)
        return query_embeds_batch

    def compute_sequence_score_batch_global(self, sequence_batch):
        sequence_batch = [self.insert_into_sequence_global(sequence,loc)[0] for sequence,loc in sequence_batch]
        sequence_batch = torch.stack(sequence_batch)
        batch_size = len(sequence_batch)
        max_token_length = (sequence_batch.shape[1] //128 + 1) *128
        product = max_token_length*batch_size
        # print(batch_size,max_token_length,product)
        max_tokens_per_sub_batch = 500000
        with torch.no_grad():
            # sequence_batch.to(self.device_ids[0])
            split =  (product // self.max_total_length)  + 1
            split_size = max(max_tokens_per_sub_batch // max_token_length, 1)
            # print('split:',split,' product:',product)
            # print('batch_size',batch_size, 'split_size',split_size,'split:',batch_size//split_size,'max_token_length',max_token_length)
            if split > 1:
                query_embeds_batch = self.split_encode(sequence_batch,split_size=split_size)
            else:
                query_embeds_batch = self.encoder(sequence_batch).last_hidden_state[:,0]


        batch_score = torch.matmul(normalize(query_embeds_batch, p=2, dim=1), self.query_list_embs.t())
        mean_batch_score = batch_score.mean(dim=1).detach().cpu()
        # early_stop = 0
        # for score in batch_score:
        #     compare_result = (score > self.clean_score).all()
        #     if compare_result == True:
        #         early_stop = 1
        #         break
        #     else:
        #         early_stop = 0

        return mean_batch_score,split
    
    def compute_sequence_score_batch(self, sequence_batch):
        sequence_batch = [self.insert_into_sequence(torch.tensor(sequence).to(self.device))[0] for sequence in sequence_batch]
        sequence_batch = torch.stack(sequence_batch)
        batch_size = len(sequence_batch)
        # sequence_lengths = torch.sum(sequence_batch!= 0, dim=1)
        # max_token_length = torch.max(sequence_lengths).item()
        max_token_length = sequence_batch.shape[1]
        # print('max_token_length',max_token_length)
        # print(batch_size ,sequence_batch.shape[1])
        product = max_token_length*batch_size
        with torch.no_grad():
            # sequence_batch.to(self.device_ids[0])
            split =  (product // self.max_total_length)  + 1
            if split > 1:
                query_embeds_batch = self.split_encode(sequence_batch,split=split)
            else:
                query_embeds_batch = self.encoder(sequence_batch).last_hidden_state[:,0]

        batch_score = torch.matmul(normalize(query_embeds_batch, p=2, dim=1), self.query_list_embs.t())
        mean_batch_score = batch_score.mean(dim=1).detach().cpu().numpy()
        early_stop = 0
        for score in batch_score:
            compare_result = (score > self.clean_score).all()
            if compare_result == True:
                early_stop = 1
                break
            else:
                early_stop = 0

        return mean_batch_score,early_stop,split

    # Function to compute gradients for a sequence
    def compute_gradients_global(self,sequence,loc):
        # sequence = torch.tensor(sequence, device=self.device)
        complete_sequence,positions = self.insert_into_sequence_global(sequence,loc) # print(len(sequence)) # 498
        onehot = torch.nn.functional.one_hot(complete_sequence.unsqueeze(0), self.vocab_size).float()
        inputs_embeds = torch.matmul(onehot, self.encoder_word_embedding)
        inputs_embeds = torch.nn.Parameter(inputs_embeds, requires_grad=True)
        # outputs = self.encoder(inputs_embeds=inputs_embeds) # [0].mean(dim=1)
        outputs = self.noise_model(inputs_embeds,loc,loc+20)

        query_embeds = outputs.last_hidden_state[:,0]
        avg_cos_sim = torch.matmul(normalize(query_embeds, p=2, dim=1), self.query_list_embs.t()).mean()

        loss = -avg_cos_sim
        loss.backward()
        gradients = inputs_embeds.grad.detach()
        # gradients = self.noise_model.noise.grad.detach()
        return gradients[0], positions  # Since batch size is 1
    
    # Function to compute gradients for a sequence
    def compute_gradients(self, sequence):
        # querys_embs_list: normarlized, List of embeddings, one for each text.
        sequence = torch.tensor(sequence, device=self.device)
        complete_sequence,positions = self.insert_into_sequence(sequence) # print(len(sequence)) # 498
        # input_ids = complete_sequence.unsqueeze(0).clone().detach().float()
        onehot = torch.nn.functional.one_hot(complete_sequence.unsqueeze(0), self.vocab_size).float()
        inputs_embeds = torch.matmul(onehot, self.encoder_word_embedding)
        inputs_embeds = torch.nn.Parameter(inputs_embeds, requires_grad=True)
        outputs = self.encoder(inputs_embeds=inputs_embeds) # [0].mean(dim=1)
        query_embeds = outputs.last_hidden_state[:,0]
        avg_cos_sim = torch.matmul(normalize(query_embeds, p=2, dim=1), self.query_list_embs.t()).mean()

        loss = avg_cos_sim
        loss.backward()
        gradients = inputs_embeds.grad.detach()
        return gradients[0],positions  # Since batch size is 1

    # Modified hotflip attack function using beam search
    def attack(self, query_list, start_tokens=None,initial_poisoned_doc=None,clean_doc=None):
        jb_score = -1
        rr_score = -1
        r_score = -1
        initial_score = -1
        jb_str = ''
        rr_str = ''
        r_str = ''
        early_stop = 0
        max_model = 'baseline'
        max_language = 'English'
        jb_poisoned_doc = None
        r_seq = None
        r_loc = None
        rr_seq = None
        rr_loc = None
        self.compute_query_embs(query_list)
        if self.jb_first == 1:
            jb_str,max_model, max_language = self.jailbreaker.gen_jailbreak_suffix(query_list, r_str,initial_poisoned_doc,self.original_pkg,self.malicious_pkg,rag_poisoned_doc=initial_poisoned_doc)
            if not jb_str is None: 
                jb_str = jb_str.text[0]
            else:
                jb_str = SUGGESTIONS_DICT['baseline']['English'].replace('malicious_pkg',self.malicious_pkg).replace('original_pkg',self.original_pkg)
            jb_poisoned_doc = self.insert_into_doc(initial_poisoned_doc,rag_str=r_str,jb_str=jb_str,replace_flag=False)
            jb_score = self.compute_doc_score(jb_poisoned_doc)

        if self.use_r == 1:
            if not jb_poisoned_doc is None:
                self.compute_doc_embs(jb_poisoned_doc,clean_doc)
            else:
                self.compute_doc_embs(initial_poisoned_doc,clean_doc)
            # Initialize beam with the initial sequence and its score
            if start_tokens is None:
                start_tokens = [0] * self.num_tokens
            start_tokens = torch.tensor(start_tokens, dtype=self.initial_poisoned_doc_sequence.dtype,device=self.device)
            initial_score = self.compute_sequence_score(start_tokens,loc=self.initial_index)
            if self.search_range == 'global':
                r_seq, r_loc, r_str, r_score,early_stop,r_str_loc = self.rank_global(start_tokens=start_tokens,initial_score=initial_score,epoch_num=self.epoch_num,initial_poisoned_doc=jb_poisoned_doc if not jb_poisoned_doc is None else initial_poisoned_doc)
            else:
                r_seq, r_str, r_score,early_stop = self.re_rank(start_tokens=start_tokens,initial_score=initial_score,epoch_num=self.epoch_num)
            r_poisoned_doc = self.insert_into_doc(initial_poisoned_doc,rag_str=r_str,replace_flag=True,str_loc=r_str_loc)

        if self.use_jb == 1 and self.jb_first == 0:
            jb_str,max_model, max_language = self.jailbreaker.gen_jailbreak_suffix(query_list, r_str,initial_poisoned_doc,self.original_pkg,self.malicious_pkg,rag_poisoned_doc=r_poisoned_doc)
            if not jb_str is None: 
                jb_str = jb_str.text[0]
            else:
                jb_str = SUGGESTIONS_DICT['baseline']['English'].replace('malicious_pkg',self.malicious_pkg).replace('original_pkg',self.original_pkg)
            
            jb_poisoned_doc = self.insert_into_doc(initial_poisoned_doc,rag_str=r_str,jb_str=jb_str,replace_flag=True,str_loc=r_str_loc)
            jb_score = self.compute_doc_score(jb_poisoned_doc)
            if self.use_rr == 1:
                jb_poisoned_doc = self.insert_into_doc(initial_poisoned_doc,rag_str=r_str,jb_str=jb_str,replace_flag=False,str_loc=r_str_loc)
                # re calculate related embeddings
                self.compute_doc_embs(jb_poisoned_doc,clean_doc,model=max_model,language=max_language)
                if self.search_range == 'global':
                    rr_seq, rr_loc, rr_str, rr_score,early_stop,rr_str_loc = self.rank_global(start_tokens=r_seq,initial_score=jb_score,epoch_num=self.rr_epoch_num,initial_poisoned_doc=jb_poisoned_doc)
                else:
                    _, rr_str, rr_score,early_stop = self.re_rank(start_tokens=r_seq,initial_score=jb_score,epoch_num=self.rr_epoch_num)

        final_poisoned_doc =  self.insert_into_doc_final(jb_poisoned_doc if (self.use_rr == 1 and self.use_jb == 1) else initial_poisoned_doc,rag_str=rr_str if (self.use_rr == 1 and self.use_jb == 1) else r_str,jb_str=jb_str,str_loc=rr_str_loc if (self.use_rr == 1 and self.use_jb == 1) else r_str_loc )
        final_score = self.compute_doc_score(final_poisoned_doc) 

        return final_poisoned_doc,r_str,jb_str,rr_str,final_score,initial_score,r_score,jb_score, rr_score,early_stop,max_model, max_language

    def rank_global(self,start_tokens,initial_score,epoch_num=50,initial_poisoned_doc=None):
        early_stop = 0

        lines = initial_poisoned_doc.splitlines()
        loc_list = []
        str_loc_list = []
        line_end_chars = []
        current_pos = 0

        def get_line_end_positions(doc):
            positions = [i for i, char in enumerate(doc) if char == '\n']
            # 处理最后一行无换行符的情况
            if doc and doc[-1] != '\n':
                positions.append(len(doc))
            return positions
        str_loc_list = get_line_end_positions(initial_poisoned_doc)
        for i,line in enumerate(lines):

            if not line:  # 处理空行
                line_end = current_pos
                line_end_chars.append(line_end - 1)  # 保持与原逻辑一致
            else:
                line_end = current_pos + len(line)
                line_end_chars.append(line_end - 1)
            current_pos = line_end + 1  # 包含换行符

        for end_char in line_end_chars:
            target_token_pos = -1
            for i, (token_start, token_end) in enumerate(self.offset_mapping):
                if token_start <= end_char < token_end:
                    target_token_pos = i
                    break
            loc_list.append(target_token_pos+1)
            
        loc_list = [(loc,str_loc) for loc,str_loc in zip(loc_list,str_loc_list) if loc != 0 ]
        initial_score_list =  [self.compute_sequence_score(start_tokens,loc=loc[0]) for loc in loc_list]
        beam = [(start_tokens,score,start_loc[0]) for start_loc,score in zip(loc_list,initial_score_list)]
        for idx,b in enumerate(beam):
            if b[2] == 272:
                beam = beam[idx:idx+1]
                break
        pbar = tqdm(ncols=150,total=epoch_num)
        final_score = 0
        counter = 0

        # create find noise model 
        # eps=8/255.
        lr=0.1

        start, end = 272,292
        complete_sequence,initial_positions = self.insert_into_sequence_global(start_tokens,start) # print(len(sequence)) # 498
        # start, end = initial_positions[0],initial_positions[-1]
        onehot = torch.nn.functional.one_hot(complete_sequence.unsqueeze(0), self.vocab_size).float()
        initial_inputs_embeds = torch.matmul(onehot, self.encoder_word_embedding)
        for s in range(start,end):
            original_inserted = initial_inputs_embeds[:, s:end, :]
            eta = original_inserted.clone()
            self.noise_model = FindNoise(self.encoder,eta).to(self.device)
            optimizer = torch.optim.Adam([{'params': self.noise_model.noise}],lr=lr)
            for epoch in range(epoch_num):
                start_time = time.time()
                all_candidates = []
                split = 0
                seq_batch = []
                for seq, score, loc in beam:
                    # Compute gradients for the current sequence
                    gradients,positions = self.compute_gradients_global(seq,loc) # (len_seq, hidden_size 768)
                    # positions = list(range(len(seq)))  # Positions to modify
                    new_seq = seq.clone()
                    # gradients_at_positions = gradients[positions]
                    # emb_grad_dotprod = torch.matmul(gradients_at_positions, self.encoder_word_embedding.t())
                    # top1tokens = torch.topk(emb_grad_dotprod, 1).indices.squeeze()
                    new_seq[:len(positions)] = seq 
                    seq_batch.append((new_seq,loc))

    
                optimizer.step()
                optimizer.zero_grad()
        
                self.encoder.zero_grad()

                beam_batch,beam_split = self.compute_sequence_score_batch_global(seq_batch)
                for (seq,loc), score in zip(seq_batch,beam_batch):
                    all_candidates.append((seq, score,loc))

                # Sort all_candidates by score in descending order and keep top beam_width sequences
                # all_candidates.sort(key=lambda x: x[1], reverse=True)
                max_triple = sorted(all_candidates, key=lambda x: x[1], reverse=True)[0]
                # beam = all_candidates[:self.beam_width]
                beam = all_candidates

                # Optionally, print the best sequence and score
                best_seq, best_score,best_loc = max_triple
                elapsed_time = time.time() - start_time 

                updated_embeds = self.noise_model.get_adv(initial_inputs_embeds,start,end)
                with torch.no_grad():    
                    outputs = self.encoder(inputs_embeds=updated_embeds) # [0].mean(dim=1)
                query_embeds = outputs.last_hidden_state[:,0]
                updated_sim = torch.matmul(normalize(query_embeds, p=2, dim=1), self.query_list_embs.t()).mean()
                error_inserted_emb = initial_inputs_embeds[:,start:end,:] + self.noise_model.noise
                emb_grad_dotprod = torch.matmul(error_inserted_emb, self.encoder_word_embedding.t())
                # 找到最相似的 token 的索引
                top1tokens = torch.topk(emb_grad_dotprod, 1).indices.squeeze()
                error_score = self.compute_sequence_score(top1tokens,start)

                pbar.set_postfix(**{'epoch':epoch,'split':split,'ini_score': initial_score,'best_score': updated_sim.item(),'err': error_score.item(),'best_loc': best_loc,'time':f'{elapsed_time:.2f}'})
                # pbar.set_postfix(**{'epoch':epoch,'split':split,'ini_score': initial_score,'best_score': final_score,'best_loc': best_loc,'time':f'{elapsed_time:.2f}'})
                pbar.update(1)

                # early_stop
                if best_score.item() > final_score:
                    final_score = best_score.item()
                    counter = 0  # 重置计数器
                else:
                    counter += 1
                    if counter >= self.patience:
                        early_stop = 1
                        # print(f'Early stopping at epoch {epoch+1}')
                        # break
        # Return the best sequence
        beam.sort(key=lambda x: x[1], reverse=True)
        best_seq, best_score,best_loc = beam[0]
        best_str = self.encoder_tokenizer.decode(best_seq)

        for loc, str_loc in loc_list:
                if loc == best_loc:
                    best_str_loc = str_loc
                    break
        return best_seq, best_loc,best_str, best_score,early_stop,best_str_loc
    


    def re_rank(self,start_tokens,initial_score,epoch_num=50):
        early_stop = 0
        final_score = 0
        counter = 0
        # if early_stop == 1: # early stop
            #     # print("early stop")
            #     # es_count += 1
            #     return self.encoder_tokenizer.decode(start_tokens),initial_poisoned_doc,initial_score.tolist(),initial_score.tolist(),early_stop
        beam = [(start_tokens, initial_score)]
        # print(f"Initial sequence score: {initial_score}")
        pbar = tqdm(ncols=150,total=epoch_num)
        for epoch in range(epoch_num):
            start_time = time.time()
            all_candidates = []
            seq_batch = []
            for seq, score in beam:
                # Compute gradients for the current sequence
                gradients,positions = self.compute_gradients(seq) # (len_seq, hidden_size 768)
                # positions = list(range(len(seq)))  # Positions to modify

                for pos_idx,pos in enumerate(positions):
                    # Get gradient at position 'pos'
                    grad_at_pos = gradients[pos]
                    # Compute dot product with embeddings
                    emb_grad_dotprod = torch.matmul(grad_at_pos, self.encoder_word_embedding.t())
                    # Get top_k_tokens tokens with highest dot product
                    topk = torch.topk(emb_grad_dotprod, self.top_k_tokens)
                    topk_tokens = topk.indices.tolist()

                    for token in topk_tokens:
                        new_seq = seq.clone()
                        new_seq[pos_idx] = token
                        # Compute score of new_seq
                        seq_batch.append(new_seq)
            score_batch,early_stop,split = self.compute_sequence_score_batch(seq_batch)

            for seq, score in zip(seq_batch, score_batch):
                all_candidates.append((seq, score))

            # Sort all_candidates by score in descending order and keep top beam_width sequences
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = all_candidates[:self.beam_width]

            # Optionally, print the best sequence and score
            best_seq, best_score = beam[0]
            # if early_stop == 1:
            #     break

            
            # print(f"Best sequence at epoch {epoch}: {self.encoder_tokenizer.decode(best_seq)} with score {best_score}")
            # tqdm.write(f"best_score {best_score}, num: {len(score_batch)}")
            elapsed_time = time.time() - start_time
            pbar.set_postfix(**{'epoch':epoch,'split':split,'best_score': best_score,'initial_score': initial_score,'time':f'{elapsed_time:.2f}'})
            pbar.update(1)

            # early_stop
            if best_score > final_score:
                final_score = best_score
                counter = 0  # 重置计数器
            else:
                counter += 1
                if counter >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        # Return the best sequence
        best_seq, best_score = beam[0]
        best_str = self.encoder_tokenizer.decode(best_seq)
        return best_seq, best_str, best_score,early_stop

    def insert_into_doc_final(self,initial_poisoned_doc,rag_str='',jb_str=None,str_loc=None):
        suggestion = SUGGESTIONS_DICT['baseline']['English'].replace('malicious_pkg',self.malicious_pkg).replace('original_pkg',self.original_pkg)
        if self.search_range == 'global':
            if not str_loc is None:
                # poisoned_seq,_ = self.insert_into_sequence_global(seq,loc)
                # inserted_doc = self.encoder_tokenizer.decode(poisoned_seq)
                inserted_doc = initial_poisoned_doc[:str_loc] + rag_str + initial_poisoned_doc[str_loc:]
        else:
            inserted_doc = initial_poisoned_doc.replace('000',f'{rag_str}',1)
               
        if not self.use_rr == 1:
            if not jb_str is None:
                inserted_doc = inserted_doc.replace(suggestion,f'{jb_str}',1)
        return inserted_doc
    
    def insert_into_doc(self,initial_poisoned_doc,rag_str='',jb_str=None,replace_flag=False,str_loc=None):
        suggestion = SUGGESTIONS_DICT['baseline']['English'].replace('malicious_pkg',self.malicious_pkg).replace('original_pkg',self.original_pkg)
        if self.search_range == 'global':
            if replace_flag:
                if not str_loc is None:
                    # poisoned_seq,_ = self.insert_into_sequence_global(seq,loc)
                    # inserted_doc = self.encoder_tokenizer.decode(poisoned_seq)
                    inserted_doc = initial_poisoned_doc[:str_loc] + rag_str + initial_poisoned_doc[str_loc:]
            else:
                inserted_doc = initial_poisoned_doc
        else:
            if self.head_insert == 1:
                if replace_flag:
                    inserted_doc = rag_str + initial_poisoned_doc
                else:
                    inserted_doc = initial_poisoned_doc
            else:
                if replace_flag:
                    inserted_doc = initial_poisoned_doc.replace('000',f'{rag_str}',1)
                else:
                    inserted_doc = initial_poisoned_doc
        if self.use_jb == 1:
            if not jb_str is None:
                inserted_doc = inserted_doc.replace(suggestion,f'{jb_str}',1)
        return inserted_doc
    
    def compute_doc_score(self,poisoned_doc):
        best_poisoned_doc_sequence = self.encoder_tokenizer(poisoned_doc, padding=False,truncation=True,return_tensors='pt').to(self.device)
        input_ids = best_poisoned_doc_sequence['input_ids'][0].clone().detach().unsqueeze(0)
        with torch.no_grad():
            outputs = self.encoder(input_ids)
        query_embeds = outputs.last_hidden_state[:,0]
        score = torch.matmul(normalize(query_embeds, p=2, dim=1), self.query_list_embs.t())
        jb_score = score.mean().detach().cpu().numpy()         
        return jb_score