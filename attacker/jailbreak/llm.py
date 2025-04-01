from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from attacker.jailbreak.sequence import Seq, MergedSeq, msg_to_seq
from attacker.jailbreak.utils import (
    ReturnStruct,
    autocast_decorator,
    compute_perplexity,
    get_nonascii_toks,
    llm_loader,
    loss_seqs,
)


class LLM(nn.Module):
    def __init__(self, params, verbose=False) -> None:
        super().__init__()
        self.params = params
        self.verbose = verbose

        self.model, self.tokenizer, self.embedding_matrix = llm_loader(
            llm_params=params.llm_params, verbose=verbose
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                # TODO: This is a hack I added because Falcon-7b-isntruct doe snot have a pad token
                # We might run into trouble here because the Seq class will automatically treat any eos_token as a pad_token and set the padding mask to 0 for this token
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self.model.device
        if self.params.allow_non_ascii:
            self.disallowed_ids = None
        else:
            self.disallowed_ids = get_nonascii_toks(self.tokenizer, device=self.device)

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path, save_embedding_layers=True)

    def model_forward(self, query_seq, use_basemodel=False):
        # reorder such that all masked tokens are on the left
        mask = query_seq.mask
        sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)

        with self.model.disable_adapter() if use_basemodel else nullcontext():
            if query_seq.is_hard:
                ids = query_seq.ids
                sorted_ids = ids.gather(1, indices)
                shifted_sorted_pred_logits = self.model(
                    input_ids=sorted_ids, attention_mask=sorted_mask
                ).logits
            else:
                embeds = query_seq.get_embed(self.embedding_matrix)
                indices_extended = indices[:, :, None].repeat(1, 1, embeds.shape[-1])
                sorted_embeds = embeds.gather(1, indices_extended)
                shifted_sorted_pred_logits = self.model(
                    inputs_embeds=sorted_embeds, attention_mask=sorted_mask
                ).logits

        # reverse the sort to get the original order (also account for the shift)
        dummy_pred_logits = torch.zeros_like(shifted_sorted_pred_logits[:, :1, :])
        sorted_pred_logits = torch.cat(
            [dummy_pred_logits, shifted_sorted_pred_logits[:, :-1, :]], dim=1
        )
        reverse_indices = indices.argsort(dim=1)
        reverse_indices_extended = reverse_indices[:, :, None].repeat(
            1, 1, sorted_pred_logits.shape[-1]
        )
        shifted_pred_logits = sorted_pred_logits.gather(1, reverse_indices_extended)
        pred_logits = torch.cat(
            [shifted_pred_logits[:, 1:, :], shifted_sorted_pred_logits[:, -1:, :]],
            dim=1,
        )

        if self.disallowed_ids is not None:
            # pred_logits[:, :, self.disallowed_ids] -= 1e10
            pred_logits[:, :, self.disallowed_ids] = -65504  # NOTE: not sure
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            for i in range(pred_logits.shape[0]):
                if torch.isnan(pred_logits[i]).any():
                    print(i, "-th logits..........", pred_logits[i])
                    print("shifted_sorted_pred_logits", shifted_sorted_pred_logits[i])
                    print("ids........", ids[i])
                    print("sorted_masks.......", sorted_mask[i])
                    print("sorted_ids", sorted_ids[i])
            raise RuntimeError(f"NaN in pred_logits: {pred_logits}")
        new_mask = torch.ones_like(mask)
        new_mask[:, :-1] = mask[:, 1:]
        seq = Seq(
            logits=pred_logits,
            mask=new_mask,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        return seq
    
    @autocast_decorator
    def compute_token_loss(self,target_tokens, **kwargs):
        gen_seqs = self.generate_teacher_forced(**kwargs)
        logits = gen_seqs.response_dist.logits
        # print(logits.shape)
        loss_batch = []
        for b in range(logits.shape[0]):
            total_loss = 0.0
            # 计算后续token的概率
            initial_probs = 1.0
            for token_pos,token in enumerate(target_tokens):
                # print(F.softmax(logits[b, token_pos], dim=-1).shape)
                initial_probs *= F.softmax(logits[b, token_pos], dim=-1)[token]
            # target_token_probs = F.softmax(logits[b, 0], dim=-1)[target_tokens]
            # print(target_token_probs)
            # total_loss -= torch.log(target_token_probs.prod(dim=-1)).sum()
            total_loss += initial_probs
            loss_batch.append(total_loss)
        
        return torch.stack(loss_batch)
    
        # if label is None:
        #     label = gen_seqs.response_teacher
        # loss_return = loss_seqs(gen_seqs.response_dist, label, **loss_params)
        # # print(loss_return.loss)
        # pred_loss_return = ReturnStruct(
        #     loss=loss_return.loss,
        #     loss_masked=loss_return.loss_masked,
        #     loss_batch=loss_return.loss_batch,
        #     query=gen_seqs.query,
        #     response_teacher=gen_seqs.response_teacher,
        #     response_dist=gen_seqs.response_dist,
        #     label=label,
        #     perplexity=gen_seqs.perplexity,
        #     perplexity_per_token_masked=gen_seqs.perplexity_per_token_masked,
        # )
        # return pred_loss_return
    
        # gen_seqs = self.generate_autoregressive(
        #     max_new_tokens=100, # cfg.train.q_params.max_new_tokens
        #     **kwargs
        # )
        # logits = gen_seqs.response_sample.logits
        # loss_batch = []

        # for b in range(logits.shape[0]):
        #     total_loss = 0.0
        #     # 检测所有可能的触发位置
        #     for pos in range(full_embeds.size(1)-1, len(input_ids.input_ids[0])-1, -1):
        #         # 检查当前token是否可能是numpy
        #         trigger_probs = F.softmax(logits[b, pos], dim=-1)[trigger_ids]
        #         if trigger_probs.max() < 0.1:
        #             continue
                
        #         # 计算后续两个token的概率
        #         next_logits = model(inputs_embeds=full_embeds[b:b+1])[:, pos+1]
        #         safe_probs = F.softmax(next_logits, dim=-1)[:, suffix_tokens]
        #         total_loss -= torch.log(safe_probs.prod(dim=-1)).sum()
        #     loss_batch.append(total_loss)


        # for batch_idx in range(logits.shape[0]):
        #     positions = target_positions
        #     total_loss = 0.0
        #     if not positions:
        #         continue  # 跳过未找到目标的样本
        #     target_token_ids = target_seq.ids[batch_idx]
        #     for pos, token_id in zip(positions, target_token_ids):
        #         if pos >= logits.shape[1]:
        #             continue
        #         logit = logits[batch_idx, pos, :]
        #         loss = F.cross_entropy(logit.unsqueeze(0), torch.tensor([token_id]).to(logit.device))
        #         total_loss += loss
        #         # valid_samples += 1
        #     loss_batch.append(total_loss)
    
        # if valid_samples == 0:
        #     return torch.tensor(0.0, requires_grad=True)
        
        # return torch.stack(loss_batch)
    
    @autocast_decorator
    def compute_pred_loss_teacher_forced(self, loss_params, label=None, **kwargs):
        gen_seqs = self.generate_teacher_forced(**kwargs)
        if label is None:
            label = gen_seqs.response_teacher
        loss_return = loss_seqs(gen_seqs.response_dist, label, **loss_params)
        # print(loss_return.loss)
        pred_loss_return = ReturnStruct(
            loss=loss_return.loss,
            loss_masked=loss_return.loss_masked,
            loss_batch=loss_return.loss_batch,
            query=gen_seqs.query,
            response_teacher=gen_seqs.response_teacher,
            response_dist=gen_seqs.response_dist,
            label=label,
            perplexity=gen_seqs.perplexity,
            perplexity_per_token_masked=gen_seqs.perplexity_per_token_masked,
        )
        return pred_loss_return

    @autocast_decorator
    def generate_teacher_forced(
        self, key, detach_query=False, use_basemodel=False, **context
    ):
        query_seq, response_teacher_seq = self.prepare_prompt(
            context, up_to_key=key, return_key_seq=True
        )
        assert not response_teacher_seq.is_empty
        full_query_seq = MergedSeq([query_seq, response_teacher_seq])

        if detach_query:
            full_query_seq = full_query_seq.clone().detach()
        pred_full_query_seq = self.model_forward(
            query_seq=full_query_seq, use_basemodel=use_basemodel
        )
        response_dist_seq = pred_full_query_seq[
            :, -response_teacher_seq.seq_len - 1 : -1
        ]
        perplexity, perplexity_per_token_masked = compute_perplexity(
            id_seq=response_teacher_seq, likelihood_seq=response_dist_seq
        )

        return_seqs = ReturnStruct(
            query=query_seq,
            response_teacher=response_teacher_seq,
            response_dist=response_dist_seq,
            perplexity=perplexity,
            perplexity_per_token_masked=perplexity_per_token_masked,
        )
        return return_seqs

    def get_next_token(self, key, use_basemodel=False, **context):
        query_seq, key_seq = self.prepare_prompt(
            context, up_to_key=key, return_key_seq=True
        )
        full_query_seq = MergedSeq([query_seq, key_seq])
        pred_dist_seq = self.model_forward(
            query_seq=full_query_seq, use_basemodel=use_basemodel
        )
        next_dist_seq = pred_dist_seq[:, -1:]

        return_seqs = ReturnStruct(query=full_query_seq, response_dist=next_dist_seq)
        return return_seqs

    def generate_autoregressive(
        self, key, use_basemodel=False, max_new_tokens=None, **context
    ):
        query_seq = self.prepare_prompt(context, up_to_key=key)
        mask = query_seq.mask
        ids = query_seq.ids

        # convert from right padding to left padding
        sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)
        sorted_ids = ids.gather(1, indices)

        generation_config = self.model.generation_config
        if self.disallowed_ids is not None:
            generation_config.suppress_tokens = self.disallowed_ids.tolist()
        generation_config.renormalize_logits = True

        if max_new_tokens is None:
            max_new_tokens = self.params.gen_params.max_new_tokens

        gen_params = dict(self.params.gen_params)
        gen_params["max_new_tokens"] = max_new_tokens

        # gen_params["eos_token_id"] = self.tokenizer.eos_token_id # new

        with self.model.disable_adapter() if use_basemodel else nullcontext():
            output = self.model.generate(
                input_ids=sorted_ids,
                attention_mask=sorted_mask,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                **gen_params,
            )

        output_ids = output.sequences[:, ids.shape[1] :]

        response_sample_seq = Seq(
            ids=output_ids, tokenizer=self.tokenizer, device=self.device
        )

        return_seqs = ReturnStruct(
            query=query_seq,
            response_sample=response_sample_seq,
        )
        return return_seqs

    def prepare_prompt(self, context, up_to_key=None, return_key_seq=False):
        seqs = []
        for msg_dct in self.params.prompt_manager.prompt_template:
            if (
                up_to_key is not None
                and up_to_key == msg_dct.key
                and not return_key_seq # False 
            ):
                break
            seq = msg_to_seq(
                msg=msg_dct.msg,
                tokenizer=self.tokenizer,
                device=self.device,
                context=context,
            )
            if up_to_key is not None and up_to_key == msg_dct.key and return_key_seq:
                break
            seqs.append(seq)

        merged_prompt_seq = MergedSeq(seqs)
        if return_key_seq:
            return merged_prompt_seq, seq
        else:
            return merged_prompt_seq
        
    def query(self,msg):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg},
        ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens=512, num_return_sequences=1, use_cache=False)
        input_length = input_ids.shape[1]
        response = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

        return response
