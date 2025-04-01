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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

# Function to disable gradient computation
def set_no_grad(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
class HotFlip:
    def __init__(self,tokenizer=None,encoder=None,hf_model=None,jailbreaker=None,cfg=None):
        # Initialize the encoder and tokenizer
        self.encoder_tokenizer = tokenizer
        self.encoder = encoder
        self.device_ids = cfg.gpu_list
        self.vocab_size = self.encoder_tokenizer.vocab_size
        self.hf_model = hf_model
        self.encoder_word_embedding = self.encoder.get_input_embeddings().weight.detach()
        # self.encoder_word_embedding = self.encoder._first_module().word_embeddings.weight

        self.encoder = DataParallel(self.encoder, device_ids=self.device_ids)
        # self.encoder.to(self.device_ids[0])
        # 初始化多进程环境
        # dist.init_process_group(backend='nccl', init_method='env://')
        # local_rank = int(os.environ['LOCAL_RANK'])
        # torch.cuda.set_device(local_rank)
        # self.encoder = DDP(self.encoder, device_ids=[local_rank], output_device=local_rank)

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
        self.original_pkg = cfg.rag.original_pkg
        self.malicious_pkg = cfg.rag.malicious_pkg
        self.jailbreaker = jailbreaker


    def compute_query_embs(self, query_list):
        self.query_list_embs = torch.tensor(self.hf_model.embed_documents(query_list),device=device) # already normarlize 

    # Function to compute document embeddings
    def compute_doc_embs(self,initial_poisoned_doc,clean_doc,model='baseline',language='English'):
        initial_poisoned_doc_token = self.encoder_tokenizer(initial_poisoned_doc, padding=False,truncation=True,return_tensors='pt', return_offsets_mapping=True).to(device)
        self.initial_poisoned_doc_sequence = initial_poisoned_doc_token['input_ids'][0]
        self.offset_mapping = initial_poisoned_doc_token["offset_mapping"][0]
        self.initial_poisoned_doc_tokenized_text = self.encoder_tokenizer.tokenize(initial_poisoned_doc,padding=False,truncation=True)

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
  
    
        self.clean_score = torch.matmul(torch.tensor(self.hf_model.embed_query(clean_doc),device=device), self.query_list_embs.t())

    # Function to compute gradient for hotflip
    def compute_hotflip_gradient(self, inputs_embeds_batch, doc_embs):
        inputs_embeds = torch.nn.Parameter(inputs_embeds_batch, requires_grad=True)
        s_adv_emb = self.encoder(inputs_embeds=inputs_embeds)[0].mean(dim=1)
        cos_sim = torch.matmul(normalize(s_adv_emb, p=2, dim=1), normalize(doc_embs, p=2, dim=1).t()).mean()
        loss = cos_sim
        loss.backward()
        return inputs_embeds.grad.detach()
    
    def replace_into_sequence(self,seq,locations):
        # 确保loc和seq为张量，并检查长度
        # loc = torch.tensor(loc, dtype=torch.long,device=original_text_tokens.device)
        # seq = torch.tensor(seq, dtype=original_text_tokens.dtype,device=original_text_tokens.device)
        assert len(locations) == len(seq), "替换位置和替换序列长度需一致"
        new_doc_tokens = self.initial_poisoned_doc_sequence.clone()

        new_doc_tokens[locations] = seq

        # # 克隆并替换
        
        # new_doc_tokens_list = [element for element in new_doc_tokens]
        # sorted_pairs = sorted(zip(seq, locations), key=lambda x: x[1], reverse=True)
        # # 按从后往前插入，避免位置偏移
        # for token_id,pos in sorted_pairs:
        #     new_doc_tokens_list.insert(pos, token_id)
        # new_doc_tokens = torch.tensor(new_doc_tokens_list,device=self.initial_poisoned_doc_input_ids['input_ids'][0].device)

        return new_doc_tokens
    
    def insert_into_sequence_global(self,inserted_tokens,pos):
        # compute original document embeddings, not query
        # insert_index = len(self.encoder_tokenizer.convert_tokens_to_ids(self.initial_poisoned_doc_tokenized_text[:pos]))
        # 合并分词结果
        combined_tokens = torch.cat([self.initial_poisoned_doc_sequence[:pos], inserted_tokens,self.initial_poisoned_doc_sequence[pos:]])
        positions = list(range(pos,pos+len(inserted_tokens)))
        # combined_tokens = original_text_tokens[:insert_index] + inserted_tokens + original_text_tokens[insert_index:]
        return combined_tokens,positions
        
    # Function to compute the score (cosine similarity) of a sequence
    def compute_sequence_score(self, sequence,loc=None):
        # querys_embs_list: normarlized, List of embeddings, one for each text.
        # sequence = torch.tensor(sequence, device=device)

        sequence,_ = self.insert_into_sequence_global(sequence,loc) # print(len(sequence)) # 498

        input_ids = sequence.unsqueeze(0)
        with torch.no_grad():
            outputs = self.encoder(input_ids)
        query_embeds = outputs.last_hidden_state[:,0]
        score = torch.matmul(normalize(query_embeds, p=2, dim=1), self.query_list_embs.t())
        mean_score = score.mean().detach().cpu().numpy()
        compare_result = (score > self.clean_score).all()
        if compare_result == True:
            early_stop = 1
        else:
            early_stop = 0
        return mean_score,early_stop

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
        max_tokens_per_sub_batch = 400000
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

    # Function to compute gradients for a sequence
    def compute_gradients_global(self,sequence,loc):
        sequence = torch.tensor(sequence, device=device)
        complete_sequence,positions = self.insert_into_sequence_global(sequence,loc) # print(len(sequence)) # 498
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
        return gradients[0], positions  # Since batch size is 1
    
    # Function to compute gradients for a sequence
    def compute_gradients(self, sequence):
        # querys_embs_list: normarlized, List of embeddings, one for each text.
        sequence = torch.tensor(sequence, device=device)
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
        t_time = time.time()
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
            start_tokens = torch.tensor(start_tokens, dtype=self.initial_poisoned_doc_sequence.dtype,device=self.initial_poisoned_doc_sequence.device)
            initial_score,early_stop = self.compute_sequence_score(start_tokens,loc=self.initial_index)
            r_seq, r_loc, r_str, r_score,early_stop,r_str_loc = self.rank_global(start_tokens=start_tokens,initial_score=initial_score,epoch_num=self.epoch_num,initial_poisoned_doc=jb_poisoned_doc if not jb_poisoned_doc is None else initial_poisoned_doc)
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
                rr_seq, rr_loc, rr_str, rr_score,early_stop,rr_str_loc = self.rank_global(start_tokens=r_seq,initial_score=jb_score,epoch_num=self.rr_epoch_num,initial_poisoned_doc=jb_poisoned_doc)
        final_poisoned_doc =  self.insert_into_doc_final(jb_poisoned_doc if (self.use_rr == 1 and self.use_jb == 1) else initial_poisoned_doc,rag_str=rr_str if (self.use_rr == 1 and self.use_jb == 1) else r_str,jb_str=jb_str,str_loc=rr_str_loc if (self.use_rr == 1 and self.use_jb == 1) else r_str_loc )
        final_score = self.compute_doc_score(final_poisoned_doc) 
        last_time = time.time() - t_time
        return final_poisoned_doc,r_str,jb_str,rr_str,final_score,initial_score,r_score,jb_score, rr_score,early_stop,max_model, max_language,last_time

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
        pbar = tqdm(ncols=120,total=epoch_num)
        final_score = 0
        counter = 0
        for epoch in range(epoch_num):
            start_time = time.time()
            all_candidates = []
            
            split = 0
            score_batch = torch.tensor([])
            beam_split = (len(beam)+9) // self.beam_width
            for beam_split_idx in range(beam_split):
                seq_batch = []
                for seq, score, loc in beam[self.beam_width*beam_split_idx:self.beam_width*(beam_split_idx+1)]:
                    # Compute gradients for the current sequence
                    gradients,positions = self.compute_gradients_global(seq,loc) # (len_seq, hidden_size 768)
                    # positions = list(range(len(seq)))  # Positions to modify
                    # print(positions[0])
                    for pos_idx,pos in enumerate(positions):
                        # Get gradient at position 'pos'
                        grad_at_pos = gradients[pos]
                        # Compute dot product with embeddings
                        emb_grad_dotprod = torch.matmul(grad_at_pos, self.encoder_word_embedding.t())
                        # (1,768)*(768,30000) = (1,30000)
                        # Get top_k_tokens tokens with highest dot product
                        topk = torch.topk(emb_grad_dotprod, self.top_k_tokens)
                        topk_tokens = topk.indices.tolist()

                        for token in topk_tokens:
                            new_seq = seq.clone()
                            new_seq[pos_idx] = token
                            # Compute score of new_seq
                            seq_batch.append((new_seq,loc))
                beam_batch,beam_split = self.compute_sequence_score_batch_global(seq_batch)
            score_batch = torch.cat((score_batch,beam_batch),dim=0)
            split = split + beam_split
            for (seq,loc), score in zip(seq_batch,score_batch):
                all_candidates.append((seq, score,loc))

            # Sort all_candidates by score in descending order and keep top beam_width sequences
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = all_candidates[:self.beam_width]

            # Optionally, print the best sequence and score
            best_seq, best_score,best_loc = beam[0]
            # if early_stop == 1:
            #     break
            elapsed_time = time.time() - start_time
            pbar.set_postfix(**{'epoch':epoch,'split':split,'ini_score':f'{initial_score:.3f}','best_score': best_score.item(),'best_loc': best_loc,'time':f'{elapsed_time:.2f}'})
            pbar.update(1)

            # early_stop
            if best_score.item() > final_score:
                final_score = best_score.item()
                counter = 0  # 重置计数器
            else:
                counter += 1
                if counter >= self.patience:
                    early_stop = 1
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        # Return the best sequence
        best_seq, best_score,best_loc = beam[0]
        best_str = self.encoder_tokenizer.decode(best_seq)

        for loc, str_loc in loc_list:
                if loc == best_loc:
                    best_str_loc = str_loc
                    break
        return best_seq, best_loc,best_str, best_score,early_stop,best_str_loc
    

    def insert_into_doc_final(self,initial_poisoned_doc,rag_str='',jb_str=None,str_loc=None):
        suggestion = SUGGESTIONS_DICT['baseline']['English'].replace('malicious_pkg',self.malicious_pkg).replace('original_pkg',self.original_pkg)

        if not str_loc is None:
            # poisoned_seq,_ = self.insert_into_sequence_global(seq,loc)
            # inserted_doc = self.encoder_tokenizer.decode(poisoned_seq)
            inserted_doc = initial_poisoned_doc[:str_loc] + rag_str + initial_poisoned_doc[str_loc:]

               
        if not self.use_rr == 1:
            if not jb_str is None:
                inserted_doc = inserted_doc.replace(suggestion,f'{jb_str}',1)
        return inserted_doc
    
    def insert_into_doc(self,initial_poisoned_doc,rag_str='',jb_str=None,replace_flag=False,str_loc=None):
        suggestion = SUGGESTIONS_DICT['baseline']['English'].replace('malicious_pkg',self.malicious_pkg).replace('original_pkg',self.original_pkg)
      
        if replace_flag:
            if not str_loc is None:
                # poisoned_seq,_ = self.insert_into_sequence_global(seq,loc)
                # inserted_doc = self.encoder_tokenizer.decode(poisoned_seq)
                inserted_doc = initial_poisoned_doc[:str_loc] + rag_str + initial_poisoned_doc[str_loc:]
        else:
            inserted_doc = initial_poisoned_doc
     
        if self.use_jb == 1:
            if not jb_str is None:
                inserted_doc = inserted_doc.replace(suggestion,f'{jb_str}',1)
        return inserted_doc
    
    def compute_doc_score(self,poisoned_doc):
        best_poisoned_doc_sequence = self.encoder_tokenizer(poisoned_doc, padding=False,truncation=True,return_tensors='pt').to(device)
        input_ids = best_poisoned_doc_sequence['input_ids'][0].clone().detach().unsqueeze(0)
        with torch.no_grad():
            outputs = self.encoder(input_ids)
        query_embeds = outputs.last_hidden_state[:,0]
        score = torch.matmul(normalize(query_embeds, p=2, dim=1), self.query_list_embs.t())
        jb_score = score.mean().detach().cpu().numpy()         
        return jb_score