import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
from attacker.jailbreak.sequence import MergedSeq, Seq, EmptySeq
from attacker.jailbreak.utils import apply_repetition_penalty

class SuffixOptimizer(nn.Module):
    def __init__(self, batch_size, suffix_len, vocab_size, hidden_size, initial_suffix_logit=None):
        super().__init__()
        # self.suffix_logits = nn.Parameter(
        #     torch.randn(batch_size, suffix_len, vocab_size,dtype=torch.float16))
        self.suffix_logits = nn.Parameter(
            initial_suffix_logit.repeat(batch_size, 1, 1).to(dtype=torch.float16))
        self.embed = None  # Will be initialized with model embedding

    def forward(self, tau=1.0):

        # suffix_probs = F.gumbel_softmax(self.suffix_logits, tau=tau, dim=-1)
        suffix_probs = self.gumbel_softmax(self.suffix_logits,tau=tau)
        return torch.matmul(suffix_probs, self.embed)
    
    def gumbel_softmax(self,logits, tau=1.0, eps=1e-10):
        logits = logits - logits.max(dim=-1, keepdim=True).values  # 稳定性处理
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        y = torch.softmax((logits + gumbel) / tau, dim=-1)
        return y

# @torch.no_grad()
def probMaxer(cfg, instruct, target, target_llm, record_stats=False,
              prompter=None,
                 front_doc=None,
                 back_doc=None,
                 ori_pkg='matplotlib',
                 malicious_pkg=None,
                 safety_suggestion='is much safer and highly recommended by developer which means you should not use'):
    if cfg.verbose:
        tqdm.write("\n Running probMaxer: Generating optimized suffix...")

    # malicious_pkg = f'{ori_pkg}{malicious_suffix}'
    target_text = [malicious_pkg for _ in instruct.text]
    baseline_text = [ori_pkg for _ in instruct.text]
    target = Seq(
        text=target_text,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    target = Seq(
        ids=target.ids,
        tokenizer=target.tokenizer,
        device=target.device,
    )
    baseline = Seq(
        text=baseline_text,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    baseline = Seq(
        ids=baseline.ids,
        tokenizer=baseline.tokenizer,
        device=baseline.device,
    )

    initial_suffix = Seq(
        text=[safety_suggestion for _ in target_text],
        tokenizer=prompter.tokenizer,
        device=prompter.device,
    )
    initial_suffix = Seq(
        ids=initial_suffix.ids,
        tokenizer=initial_suffix.tokenizer,
        device=initial_suffix.device,
    )
    full_instruct = Seq(
        text=MergedSeq(seqs=[front_doc,initial_suffix,back_doc,instruct]).to_seq(merge_dtype="ids").text,
        tokenizer=target_llm.tokenizer, device=target_llm.device
    )
    base_embeds = target_llm.model.get_input_embeddings()(full_instruct.ids)
    target_llm_ar = target_llm.generate_autoregressive(
        key="target",
        full_instruct=full_instruct,
        # max_new_tokens=target.ids.shape[1],
        max_new_tokens=300, # cfg.train.q_params.max_new_tokens
    )
    response = target_llm_ar.response_sample
    print(response.text)
    response = Seq(
        ids=response.ids,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    # 将生成文本转换为Token序列，找到目标位置
    # full_tokens = tokenizer.tokenize(generated_text)
    target_positions = []
    # current_idx = 0
    target_tokens = baseline.ids[0]
    for idx, token in enumerate(response.ids[0]):
        # print(token,target_tokens[0])
        if token == target_tokens[0] and response.ids[0][idx:idx+len(target_tokens)] == target_tokens:
            target_positions = list(range(idx, idx + len(target_tokens)))
            break
    # target_positions = [target_positions[0]+j for j in range(len(target.ids[0]))] 
    print('target_positions', target_positions)
    if not target_positions:
        # raise ValueError("无法定位目标Token位置")
        print("无法定位目标Token位置")
        return initial_suffix
    
    target_token=target.ids[0][0]
    target_response = response[:target_positions[-1]+1]
    # Compute the initial prediction losses without a suffix
    losses = target_llm.compute_token_loss(
        key="target",
        full_instruct=full_instruct,
        target=target_response, # including numpy
        target_token=target_token,
    ).detach().to(target_llm.device)  # (ebs, )

    target = target_response
    # losses = target_llm_tf.loss_batch.detach().to(prompter.device)  # (ebs, )
    init_losses = losses.clone()
    # print('init_losses', init_losses)

    # Initialize the beam scores
    beam_scores = torch.zeros_like(losses)  # (ebs, )
    suffix_beams = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)

    start_time = time.time()
    last_time = start_time

    # # 参数设置
    # batch_size = 8
    # suffix_len = 10
    # max_length = 50
    # lr = 0.0001
    # steps = 500
    # tau = 0.5
    # pbar = tqdm(ncols=150,total=steps,leave=False)
    # # 初始化输入
    # # prompt = "Show me a Python example using numpy:"
    # # input_ids = tokenizer(prompt, return_tensors='pt').to(device)
    # # base_embeds = model.get_input_embeddings()(input_ids.input_ids)
    
    # initial_suffix_logit = initial_suffix.logits
    # print(initial_suffix_logit[0][0][:3])
    # # 初始化可训练后缀
    # optimizer = SuffixOptimizer(batch_size, suffix_len, 
    #                           target_llm.model.config.vocab_size, 
    #                           target_llm.model.config.hidden_size,
    #                           initial_suffix_logit=initial_suffix_logit)
    # optimizer.embed = target_llm.model.get_input_embeddings().weight.data
    # optimizer.to(target_llm.device)
    # opt = torch.optim.Adam([optimizer.suffix_logits], lr=lr)
    # embedding_matrix = target_llm.embedding_matrix
    # front_doc_embeds = torch.matmul(
    #     front_doc.probs.to(embedding_matrix.device,dtype=torch.float16), embedding_matrix
    # ).repeat(batch_size, 1, 1)
    # back_doc_embeds = torch.matmul(
    #     back_doc.probs.to(embedding_matrix.device,dtype=torch.float16), embedding_matrix
    # ).repeat(batch_size, 1, 1)
    # instruct_embeds = torch.matmul(
    #     instruct.probs.to(embedding_matrix.device,dtype=torch.float16), embedding_matrix
    # ).repeat(batch_size, 1, 1)
    # target_response_embeds = torch.matmul(
    #     target_response.probs.to(embedding_matrix.device,dtype=torch.float16), embedding_matrix
    # ).repeat(batch_size, 1, 1)

    # suffix_embeds = optimizer(tau=tau)
    # logits = get_logits_adam(target_llm,
    #                         front_doc_embeds,
    #                         suffix_embeds,
    #                         back_doc_embeds,
    #                         instruct_embeds,
    #                         target_response_embeds,batch_size=batch_size)
    # loss = 0.0
    # for b in range(logits.shape[0]):
    #     target_token_probs = F.softmax(logits[b, 0], dim=-1)[target_token]
    #     loss -= torch.log(target_token_probs.prod(dim=-1)).sum()
    # init_loss = loss
    # # 训练循环
    # for step in range(steps):
    #     t_time = time.time()
    #     opt.zero_grad()
    #     # 生成后缀嵌入
    #     suffix_embeds = optimizer(tau=tau)
    #     print(optimizer.suffix_logits[0][0][:3])
    #     logits = get_logits_adam(target_llm,
    #                              front_doc_embeds,
    #                             suffix_embeds,
    #                             back_doc_embeds,
    #                             instruct_embeds,
    #                             target_response_embeds,batch_size=batch_size)
    #     print(logits[0,0,0])
    #     loss = 0.0
    #     for b in range(logits.shape[0]):
    #         target_token_probs = F.softmax(logits[b, 0], dim=-1)[target_token]
    #         print(target_token_probs)
    #         loss -= torch.log(target_token_probs.prod(dim=-1)).sum()
    #     # 反向传播
    #     loss.backward()
    #     # print(loss.grad)
    #     # print(logits.grad[0,0,0])
    #     opt.step()
        
    #     # if step % 50 == 0:
    #     #     print(f"Step {step}, Loss: {loss.item()}")

    #     last_time = time.time() - t_time
    #     pbar.set_postfix(**{'step':step,'last': f'{last_time:.2f}s','ini': f'{init_loss.item():.2f}','loss': f'{loss.item():.2f}'})
    #     pbar.update(1)

    # # 获取最终后缀
    # suffix_tokens = F.gumbel_softmax(optimizer.suffix_logits, hard=True).argmax(-1)
    # optimized_suffix = target_llm.tokenizer.batch_decode(suffix_tokens)
    # print(optimized_suffix)
    # return optimized_suffix

    pbar = tqdm(ncols=150,total=cfg.train.q_params.max_new_tokens,leave=False)
    for idx in range(cfg.train.q_params.max_new_tokens):
        t_time = time.time()
        if idx == 0:
            num_beams_in = 1
            num_beams_out = cfg.train.q_params.num_beams
        elif idx == cfg.train.q_params.max_new_tokens - 1:
            num_beams_in = cfg.train.q_params.num_beams
            num_beams_out = 1
        else:
            num_beams_in = cfg.train.q_params.num_beams
            num_beams_out = cfg.train.q_params.num_beams

        # expand the dimension of instruct and targets to match suffix beams
        instruct_rep = instruct.repeat_interleave(num_beams_in, dim=0)
        target_rep = target.repeat_interleave(num_beams_in, dim=0)

        next_dist_seq_prompter, next_dist_seq_basemodel = get_next_token_probabilities(
            cfg=cfg, instruct=instruct_rep, suffix=suffix_beams, prompter=prompter, 
            front_doc=front_doc,back_doc=back_doc,
        )

        next_token_candidate_ids, candidate_beam_scores, candidate_losses = (
            select_and_evaluate_next_token_candidates(
                cfg=cfg,
                instruct=instruct_rep,
                target=target_rep,
                suffix=suffix_beams,
                target_llm=target_llm,
                next_dist_seq_prompter=next_dist_seq_prompter,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                beam_scores=beam_scores,
                prev_losses=losses,
                num_beams_in=num_beams_in,
                front_doc=front_doc,back_doc=back_doc,
                target_token=target_token,
            )
        )

        prev_losses = losses
        suffix_beams, losses, beam_scores = select_next_beams(
            cfg=cfg,
            suffix_beams=suffix_beams,
            next_token_candidate_ids=next_token_candidate_ids,
            candidate_beam_scores=candidate_beam_scores,
            candidate_losses=candidate_losses,
            num_beams_in=num_beams_in,
            num_beams_out=num_beams_out,
        )

        # early stopping
        loss_diff = (
            losses.reshape(-1, num_beams_out)
            - prev_losses.reshape(-1, num_beams_in).min(-1).values[:, None]
        )
        min_loss_diff = loss_diff.min(-1).values.item()  # of all beams
        stopped = min_loss_diff > 0
        # tqdm.write(f"Score diff: {min_loss_diff}")

        if cfg.verbose:
            tqdm.write(f" Beams[0] (iter {idx}): {suffix_beams[:num_beams_out].text}")

        last_time = time.time() - t_time
        # tot_time = time.time() - start_time
        pbar.set_postfix(**{'token':idx,'suffix':suffix_beams[0].text,'last': f'{last_time:.2f}s','ini_loss': f'{init_losses[0].item():.2f}','loss': f'{losses[0].item():.2f}'})
        # pbar.set_postfix(**{'Score diff': f'{min_loss_diff:.4f}s'})
        pbar.update(1)

        # if stopped:
        #     break

    if cfg.verbose:
        tqdm.write(
            f" AdvPrompterOpt completed. Generated suffix[0]: {suffix_beams[0].text}"
        )

    if record_stats:
        return suffix_beams, dict(
            init_loss=torch.tensor(init_losses), loss=torch.tensor(losses)
        )
    return suffix_beams


def get_next_token_probabilities(cfg, instruct, suffix, prompter,front_doc,back_doc):
    # get the next token probabilities from the prompter and the base model
    prompter_next = prompter.get_next_token(
        key="suffix", instruct=instruct, suffix=suffix, front_doc=front_doc,back_doc=back_doc
    )
    next_dist_seq_prompter = (
        prompter_next.response_dist.clone().detach()
    )  # (ebs, 1, vocab_size)
    prompter_next_basemodel = prompter.get_next_token(
        key="suffix", instruct=instruct, suffix=suffix, use_basemodel=True,front_doc=front_doc,back_doc=back_doc,
    )
    next_dist_seq_basemodel = (
        prompter_next_basemodel.response_dist.clone().detach()
    )  # (bs, 1, vocab_size)

    # apply repetition penalty
    if not suffix.is_empty and "repetition_penalty" in cfg.train.q_params:
        next_dist_logits_basemodel = apply_repetition_penalty(
            logits=next_dist_seq_basemodel.logits.squeeze(1),
            prev_ids=suffix.ids,
            penalty=cfg.train.q_params.repetition_penalty,
        )
        next_dist_seq_basemodel = Seq(
            logits=next_dist_logits_basemodel[:, None, :],
            mask=next_dist_seq_basemodel.mask,
            tokenizer=next_dist_seq_basemodel.tokenizer,
            device=next_dist_seq_basemodel.device,
        )
        next_dist_logits_prompter = apply_repetition_penalty(
            logits=next_dist_seq_prompter.logits.squeeze(1),
            prev_ids=suffix.ids,
            penalty=cfg.train.q_params.repetition_penalty,
        )
        next_dist_seq_prompter = Seq(
            logits=next_dist_logits_prompter[:, None, :],
            mask=next_dist_seq_prompter.mask,
            tokenizer=next_dist_seq_prompter.tokenizer,
            device=next_dist_seq_prompter.device,
        )
    return next_dist_seq_prompter, next_dist_seq_basemodel


def select_and_evaluate_next_token_candidates(
    cfg,
    instruct,
    target,
    suffix,
    target_llm,
    next_dist_seq_prompter,
    next_dist_seq_basemodel,
    beam_scores,
    prev_losses,
    num_beams_in,
    front_doc,
    back_doc,
    target_token,
):
    num_chunks = cfg.train.q_params.num_chunks
    assert cfg.train.q_params.top_k % (num_chunks * num_beams_in) == 0
    num_samples_per_beam = cfg.train.q_params.top_k // (num_chunks * num_beams_in)
    all_next_token_candidate_ids = None

    for i in range(cfg.train.q_params.num_chunks):
        next_token_candidate_ids = select_next_token_candidates(
            cfg=cfg,
            next_dist_seq=next_dist_seq_prompter,
            previous_next_token_candidate_ids=all_next_token_candidate_ids,
            num_samples_per_beam=num_samples_per_beam,
            always_include_best=cfg.train.q_params.candidates.always_include_best
            and i == 0,
        )  # (ebs = bs * num_beams_in, num_samples_per_beam)

        candidate_beam_scores, candidate_losses = evaluate_next_token_candidates(
            cfg=cfg,
            instruct=instruct,
            target=target,
            suffix=suffix,
            target_llm=target_llm,
            next_token_candidate_ids=next_token_candidate_ids,
            next_dist_seq_basemodel=next_dist_seq_basemodel,
            next_dist_seq_prompter=next_dist_seq_prompter,
            prev_beam_scores=beam_scores,
            prev_losses=prev_losses,
            front_doc=front_doc,back_doc=back_doc,
            target_token=target_token,
        )  # (ebs, num_samples_per_beam)

        if all_next_token_candidate_ids is None:
            all_next_token_candidate_ids = next_token_candidate_ids
            all_candidate_beam_scores = candidate_beam_scores
            all_candidate_losses = candidate_losses
        else:
            all_next_token_candidate_ids = torch.cat(
                (next_token_candidate_ids, all_next_token_candidate_ids), dim=1
            )  # (ebs, i * num_samples_per_beam)
            all_candidate_beam_scores = torch.cat(
                (candidate_beam_scores, all_candidate_beam_scores), dim=1
            )  # (ebs, i * num_samples_per_beam)
            all_candidate_losses = torch.cat(
                (candidate_losses, all_candidate_losses), dim=1
            )  # (ebs, i * num_samples_per_beam)
    return all_next_token_candidate_ids, all_candidate_beam_scores, all_candidate_losses


@torch.no_grad()
def select_next_token_candidates(
    cfg,
    next_dist_seq,
    previous_next_token_candidate_ids,
    num_samples_per_beam,
    always_include_best,
):
    # clone is important here! We modify the logits but will also use the original dist
    next_dist_logits = next_dist_seq.logits.squeeze(1).clone()  # (ebs, vocab_size)

    if previous_next_token_candidate_ids is not None:
        previous_next_token_candidate_ids_khot = torch.scatter(
            torch.zeros_like(next_dist_logits), 1, previous_next_token_candidate_ids, 1
        )  # (ebs, vocab_size)
        next_dist_logits -= 1e10 * previous_next_token_candidate_ids_khot

    if cfg.train.q_params.candidates.do_sample:
        if always_include_best:
            next_dist_logits -= 1e10 * next_dist_seq.onehot.squeeze(1)

        probs = torch.softmax(
            next_dist_logits / cfg.train.q_params.candidates.temperature,
            dim=-1,
        )  # (ebs, vocab_size)
        next_token_candidate_ids = probs.multinomial(
            num_samples=num_samples_per_beam, replacement=False
        )  # (ebs, num_samples_per_beam)
        if always_include_best:
            next_token_candidate_ids = torch.cat(
                [next_dist_seq.ids, next_token_candidate_ids[:, :-1]], dim=1
            )
    else:
        next_token_candidate_ids = next_dist_logits.topk(
            k=num_samples_per_beam, dim=-1
        ).indices  # (ebs, num_samples_per_beam)
    return next_token_candidate_ids


@torch.no_grad()
def evaluate_next_token_candidates(
    cfg,
    instruct,
    target,
    suffix,
    target_llm,
    next_token_candidate_ids,
    next_dist_seq_basemodel,
    next_dist_seq_prompter,
    prev_beam_scores,
    prev_losses,
    front_doc,
    back_doc,
    target_token,
):
    ebs, num_samples_per_beam = next_token_candidate_ids.shape

    q_next_token_candidate_ids = torch.reshape(
        next_token_candidate_ids, (ebs * num_samples_per_beam, 1)
    )
    q_sample_seq = Seq(
        ids=q_next_token_candidate_ids,
        tokenizer=next_dist_seq_prompter.tokenizer,
        device=next_dist_seq_prompter.device,
    )

    # extend to match the extended batch size
    instruct_rep = instruct.repeat_interleave(num_samples_per_beam, dim=0)
    target_rep = target.repeat_interleave(num_samples_per_beam, dim=0)
    # target_positions_rep=[target_positions for _ in range(num_samples_per_beam) ]
    if not suffix.is_empty:
        suffix_rep = suffix.repeat_interleave(num_samples_per_beam, dim=0)
    else:
        suffix_rep = suffix

    # compute the losses on each sample
    merged=MergedSeq(seqs=[front_doc,suffix_rep,back_doc,instruct_rep])
    # merged = MergedSeq(seqs=[instruct_rep, suffix_rep, q_sample_seq])
    full_instruct = Seq(
        text=merged.to_seq(merge_dtype="ids").text,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )

    with torch.no_grad():
        # target_llm_tf_q = target_llm.compute_pred_loss_teacher_forced(
        #     key="target",
        #     full_instruct=full_instruct,
        #     target=target_rep,
        #     loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        # )
        target_llm_tf_q = target_llm.compute_token_loss(
            key="target",
            full_instruct=full_instruct,
            target=target_rep, # including numpy
            target_token=target_token,
            # loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        )

    # loss_batch = target_llm_tf_q.loss_batch.to(next_dist_seq_prompter.device)
    loss_batch = target_llm_tf_q.to(next_dist_seq_prompter.device)
    losses = torch.reshape(loss_batch, (ebs, num_samples_per_beam))
    loss_delta = losses - prev_losses[:, None]  # (ebs, num_samples_per_beam)
    # tqdm.write(f"# of improved beams: {(loss_delta < 0).sum().item()}")
    # num_improved_beams = (loss_delta < 0).sum().item()
    next_dist_logprobs_basemodel = next_dist_seq_basemodel.logprobs.squeeze(1)
    selected_logprobs_basemodel = torch.gather(
        next_dist_logprobs_basemodel, dim=-1, index=next_token_candidate_ids
    )  # (ebs, num_samples_per_beam)

    factor = cfg.train.q_params.lambda_val
    beam_scores_delta = selected_logprobs_basemodel - loss_delta * factor
    new_beam_scores = prev_beam_scores[:, None] + beam_scores_delta
    return new_beam_scores, losses


@torch.no_grad()
def calculate_reward(cfg, prompter, target_llm, instruct, suffix, target):
    target_llm_instruct = Seq(
        text=MergedSeq(seqs=[instruct, suffix]).to_seq(merge_dtype="ids").text,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    target_llm_target = target

    prompter_instruct = instruct
    prompter_suffix = suffix
    prompter_target = Seq(
        text=target.text, tokenizer=prompter.tokenizer, device=prompter.device
    )

    target_llm_losses = (
        target_llm.compute_pred_loss_teacher_forced(
            key="target",
            full_instruct=target_llm_instruct,
            target=target_llm_target,
            loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        )
        .loss_batch.detach()
        .cpu()
    )
    prompter_losses = (
        prompter.compute_pred_loss_teacher_forced(
            key="target",
            instruct=prompter_instruct,
            suffix=prompter_suffix,
            target=prompter_target,
            loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        )
        .loss_batch.detach()
        .cpu()
    )

    return prompter_losses - cfg.train.q_params.lambda_val * target_llm_losses


@torch.no_grad()
def select_next_beams(
    cfg,
    suffix_beams,
    next_token_candidate_ids,
    candidate_beam_scores,
    candidate_losses,
    num_beams_in,
    num_beams_out,
):
    ebs, num_samples_per_beam = candidate_beam_scores.shape
    bs = ebs // num_beams_in

    candidate_beam_scores = candidate_beam_scores.reshape(
        bs, num_beams_in * num_samples_per_beam
    )  # (bs, num_beams_in * num_samples_per_beam)
    if cfg.train.q_params.beams.do_sample:
        if cfg.train.q_params.beams.always_include_best:
            candidate_beam_scores_top_ids = candidate_beam_scores.argmax(dim=-1)
            candidate_beam_scores_onehot = torch.zeros_like(candidate_beam_scores)
            candidate_beam_scores_onehot.scatter_(
                1, candidate_beam_scores_top_ids[:, None], 1
            )
            candidate_beam_scores_corrected = (
                candidate_beam_scores - 1e10 * candidate_beam_scores_onehot
            )
            beam_probs = torch.softmax(
                candidate_beam_scores_corrected / cfg.train.q_params.beams.temperature,
                dim=-1,
            )  # (bs, num_beams_in * num_samples_per_beam)
        else:
            beam_probs = torch.softmax(
                candidate_beam_scores / cfg.train.q_params.beams.temperature,
                dim=-1,
            )  # (bs, num_beams_in * num_samples_per_beam)
        next_beam_indices = beam_probs.multinomial(
            num_samples=num_beams_out, replacement=False
        )  # (bs, num_beams_out) [0, num_beams_in * num_samples_per_beam]
        if cfg.train.q_params.beams.always_include_best:
            next_beam_indices = torch.cat(
                [candidate_beam_scores_top_ids[:, None], next_beam_indices[:, :-1]],
                dim=-1,
            )
    else:
        next_beam_indices = candidate_beam_scores.topk(
            k=num_beams_out, dim=-1, sorted=True
        ).indices  # (bs, num_beams_out) [0, num_beams_in * num_samples_per_beam]

    next_beam_indices_expanded = (
        next_beam_indices.cpu()
        + torch.arange(0, bs)[:, None] * num_beams_in * num_samples_per_beam
    )  # (bs, num_beams_out)
    next_beam_indices_expanded = next_beam_indices_expanded.reshape(-1)

    next_token_candidate_seq = Seq(
        ids=next_token_candidate_ids.reshape(
            bs * num_beams_in * num_samples_per_beam, 1
        ),
        tokenizer=suffix_beams.tokenizer,
        device=suffix_beams.device,
    )

    if suffix_beams.is_empty:
        next_suffix_beams = next_token_candidate_seq[next_beam_indices_expanded]
    else:
        beam_candidates = suffix_beams.repeat_interleave(num_samples_per_beam, dim=0)
        beam_candidates.append(next_token_candidate_seq)
        next_suffix_beams = beam_candidates[next_beam_indices_expanded]

    candidate_losses = candidate_losses.reshape(
        bs, num_beams_in * num_samples_per_beam
    )  # (bs, num_beams_in * num_samples_per_beam)
    selected_losses = candidate_losses.gather(
        dim=1, index=next_beam_indices
    )  # (bs, num_beams_out)
    selected_losses = selected_losses.reshape(bs * num_beams_out).detach()

    selected_beam_scores = candidate_beam_scores.gather(
        dim=1, index=next_beam_indices
    )  # (bs, num_beams_out)
    selected_beam_scores = selected_beam_scores.reshape(bs * num_beams_out).detach()
    return next_suffix_beams, selected_losses, selected_beam_scores

def get_logits_adam(target_llm,front_doc_embeds,suffix_embeds,back_doc_embeds, instruct_embeds,target_response_embeds,batch_size=8):
    # 拼接输入
    full_embeds = torch.cat([front_doc_embeds,suffix_embeds,back_doc_embeds, instruct_embeds,target_response_embeds], dim=1)
    # mask = query_seq.mask
    mask = torch.ones((batch_size, full_embeds.shape[1]), device=target_llm.device).bool()
    sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)

    # with target_llm.model.disable_adapter():
        # embeds = query_seq.get_embed(embedding_matrix)
    embeds = full_embeds
    indices_extended = indices[:, :, None].repeat(1, 1, embeds.shape[-1])
    sorted_embeds = embeds.gather(1, indices_extended)
    shifted_sorted_pred_logits = target_llm.model(
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
    if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
        for i in range(pred_logits.shape[0]):
            if torch.isnan(pred_logits[i]).any():
                print(i, "-th logits..........", pred_logits[i])
                print("shifted_sorted_pred_logits", shifted_sorted_pred_logits[i])
                print("sorted_masks.......", sorted_mask[i])
        raise RuntimeError(f"NaN in pred_logits: {pred_logits}")
    return pred_logits