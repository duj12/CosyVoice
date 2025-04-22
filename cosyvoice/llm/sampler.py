import torch.nn as nn
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch

from sgl_kernel import (
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
)

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits, #[1,NUM_CALSSES]
        temperature,
        top_k,
        top_p,
    ):

        # Post process logits
        logits.div_(torch.FloatTensor([[temperature]]).to(logits.device)) #[1,1]
        probs = torch.softmax(logits, dim=-1)

        max_top_k_round, batch_size = 32, probs.shape[0]
        uniform_samples = torch.rand(
            (max_top_k_round, batch_size), device=probs.device
        )
        batch_next_token_ids, success = top_k_top_p_sampling_from_probs(
            probs,
            uniform_samples,
            torch.LongTensor([top_k]).to(logits.device).repeat(batch_size), #[1]
            torch.FloatTensor([top_p]).to(logits.device).repeat(batch_size), #[1]
            filter_apply_order="joint",
        )
        return batch_next_token_ids.to(torch.int32)

sampler = Sampler()
# sampler = torch.compile(sampler,mode='reduce-overhead')

import time
def ras_sampling2(weighted_scores_in, decoded_tokens_in, top_p=0.8, top_k=25, temperature=1.0, win_size=10, tau_r=0.1,resample={'topp': 0.15,'topk':80,'temperature':1.2}):
    # weigted_scores_in: [B, Num_classes]
    # print("ras debug", weighted_scores_in.size(), decoded_tokens_in.size(), end=" ")
    decoded_tokens = decoded_tokens_in  # [0, ...]

    # st = time.perf_counter()
    top_ids = sampler(weighted_scores_in, temperature, top_p=top_p, top_k=top_k)#[B]
    # et = time.perf_counter()
    # print("sampling time:", et-st)
    rep_nums = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores_in.device) == top_ids).sum(0)
    
    
    # st = time.perf_counter()
    rep_top_ids = sampler(weighted_scores_in, resample['temperature'], top_p=top_p+resample['topp'], top_k=top_k+resample['topk'])
    top_ids = torch.where(rep_nums >= win_size * tau_r, rep_top_ids, top_ids)
    # for idx in range(rep_nums.shape[0]):
    #     if rep_nums[idx] >= win_size * tau_r:
    #         top_ids[idx] = rep_top_ids[idx]
    #         # top_ids[idx] = sampler(weighted_scores_in[idx:idx+1], resample['temperature'], top_p=top_p+resample['topp'], top_k=top_k+resample['topk']).item()
    # et = time.perf_counter()
    # print("resample time:", et-st)
    return top_ids

if __name__ == "__main__":
    logits = torch.randn(1, 1000).cuda()
    sampler = Sampler()
    result = sampler.forward(logits, 1.0, 0, 0.9)
    pass