#!/usr/bin/env python3
# -*- coding: utf-8 -*-"""
"""

Reranker / Sentence Extractor / Reader Model

Used for both 1st stage para/sentence reranking and 2nd stage sentence reranking 

Inspired by https://github.com/stanford-futuredata/Baleen
Code adapted from https://github.com/facebookresearch/multihop_dense_retrieval


@author Tim Hartill

"""
import numpy as np
from transformers import AutoModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F

from mdr_basic_tokenizer_and_utils import get_final_text



class EvidenceCombiner(nn.Module):
    """ Input: cls [bs,hs] , insuff ans [bs,hs] , sent markers [bs, max#sentsinbatch, hs] 
        Output: [bs, 2]
    """
    def __init__(self, config, dropout_prob=0.0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.dense1 = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, cls_hs, insuff_hs, sent_hs):
        sent_mean_hs = sent_hs.mean(dim=1)      # [bs, hs] Note where batch["sent_offsets"] = 0 eg from padding will have gathered cls 
        hidden_states = torch.cat([cls_hs, insuff_hs, sent_mean_hs], dim=1)    # [bs, hs * 3]
        out = F.gelu(self.dense1(hidden_states))
        out = self.dense2(out)
        out = self.dropout(out)
        return out


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class StageModel(nn.Module):

    def __init__(self, config, args):
        super().__init__()
        if args.__dict__.get('model_name_stage') is not None:
            self.model_name = args.model_name_stage
        else:    
            self.model_name = args.model_name
        self.sp_weight = args.sp_weight
        self.debug = args.debug
        self.debug_count = 3
        self.sent_score_force_zero = args.sent_score_force_zero
        self.ev_combiner = args.ev_combiner
        self.encoder = AutoModel.from_pretrained(self.model_name)

        if "electra" in self.model_name:
            self.pooler = BertPooler(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.rank = nn.Linear(config.hidden_size, 1) 

        self.sp = nn.Linear(config.hidden_size, 1)

        if self.ev_combiner:        
            self.ev_combiner_head = EvidenceCombiner(config)
        
        self.loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")
        if self.ev_combiner:
            self.ev_loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, batch):

        outputs = self.encoder(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids', None))

        if "electra" in self.model_name:
            sequence_output = outputs[0]  # [0]=raw seq output [bs, seq_len, hs]
            pooled_output = self.pooler(sequence_output) # [bs, hs]
        else:
            sequence_output, pooled_output = outputs[0], outputs[1]

        logits = self.qa_outputs(sequence_output) # [bs, seq_len, 2]
        outs = [o.squeeze(-1) for o in logits.split(1, dim=-1)]  # [ [bs, seq_len], [bs, seq_len] ]
        #TJH could remove para mask but would need to adjust 'doc_tokens' to include everything new. originally fill everything not in a para with -inf:  [ [bs, seq_len], [bs, seq_len] ]
        outs = [o.float().masked_fill(batch["paragraph_mask"].ne(1), float("-inf")).type_as(o) for o in outs]  #TJH ne = elementwise not equal

        start_logits, end_logits = outs[0], outs[1]  # start_logits: [bs, seq_len]  end_logits: [bs, seq_len]
        rank_score = self.rank(pooled_output)  # [bs, 1]
        # batch['sent_offsets'] = [bs, max#sentsinbatch]. Note if no sent markers or in padding will repeatedly gather seq 0=cls
        gather_index = batch["sent_offsets"].unsqueeze(2).expand(-1, -1, sequence_output.size()[-1])  # [bs, max#sentsinbatch, hs]
        sent_marker_rep = torch.gather(sequence_output, 1, gather_index)  # [bs, max#sentsinbatch, hs] gather along seq_len of [bs, seq_len, hs]
        sp_score = self.sp(sent_marker_rep).squeeze(2)  # [bs, #sents, 1] -> [bs, #sents]
        
        
        if self.ev_combiner:
            insuff_hs = torch.cat([sequence_output[i, idx].unsqueeze(0) for i, idx in enumerate(batch["insuff_offset"].squeeze(1))], dim=0)
            ev_logits = self.ev_combiner_head(sequence_output[:,0], insuff_hs, sent_marker_rep)  # [bs,2]

        if self.training:

            rank_target = batch["label"]
            rank_loss = F.binary_cross_entropy_with_logits(rank_score, rank_target.float(), reduction="sum")
            
            if self.ev_combiner:
                ev_loss = self.ev_loss_fct(ev_logits , rank_target.squeeze(1))

            #batch["sent_labels"] = [bs, max#sentsinbatch]
            sp_loss = F.binary_cross_entropy_with_logits(sp_score, batch["sent_labels"].float(), reduction="none")  # [bs, max#sentsinbatch]
            if self.sent_score_force_zero:
                sent_mask = (batch["sent_offsets"] != 0).type(torch.int64)  #[bs, max#sentsinbatch] padding = 0, actual sent offset = 1
                sp_loss = (sp_loss * sent_mask)  
                #sp_loss = (sp_loss * batch["sent_offsets"]) * batch["label"]  # batch["sent_offsets"] padded with zeros so effectively zeros the loss for padding...
            sp_loss = sp_loss.sum()

            start_positions, end_positions = batch["starts"], batch["ends"]  
            # torch.unbind converts [ [1], [2], [3] ] to ([1,2,3]) like .squeeze(-1) only in tuple
            start_losses = [self.loss_fct(start_logits, starts) for starts in torch.unbind(start_positions, dim=1)]
            end_losses = [self.loss_fct(end_logits, ends) for ends in torch.unbind(end_positions, dim=1)]
            loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)
            log_prob = - loss_tensor
            log_prob = log_prob.float().masked_fill(log_prob == 0, float('-inf')).type_as(log_prob)
            marginal_probs = torch.sum(torch.exp(log_prob), dim=1)
            m_prob = [marginal_probs[idx] for idx in marginal_probs.nonzero()]
            if len(m_prob) == 0:
                span_loss = self.loss_fct(start_logits, start_logits.new_zeros(start_logits.size(0)).long()-1).sum()
            else:
                span_loss = - torch.log(torch.cat(m_prob)).sum()
            if self.ev_combiner:
                loss = ev_loss + rank_loss + span_loss + sp_loss * self.sp_weight
            else:
                loss = rank_loss + span_loss + sp_loss * self.sp_weight
                
            if self.debug and self.debug_count > 0:
                print(f"LOSSES: rank_loss:{rank_loss}  span_loss:{span_loss}  sp_loss_before_weight:{sp_loss}  sp_weight:{self.sp_weight}")
                self.debug_count -= 1
            return loss.unsqueeze(0)
        
        if self.ev_combiner:
            return {
                'start_logits': start_logits,   # [bs, seq_len]
                'end_logits': end_logits,       # [bs, seq_len]
                'rank_score': rank_score,       # [bs,1] is para evidential 0<->1
                'ev_logits': ev_logits,         # [bs, 2] logit 0=not evidential/logit 1=evidential para/context evidential considering cls, insuff and sent marker hidden states
                "sp_score": sp_score            # [bs, num_sentences] is sentence evidential [0,1,0,0..]
                }
        else:
            return {
                'start_logits': start_logits,   # [bs, seq_len]
                'end_logits': end_logits,       # [bs, seq_len]
                'rank_score': rank_score,       # [bs,1] is para evidential 0<->1
                "sp_score": sp_score            # [bs, num_sentences] is sentence evidential [0,1,0,0..]
                }


class SpanAnswerer():
    def __init__(self, item_list, outs, para_offset, insuff_offset, max_ans_len=35):
        if type(para_offset) == list:
            para_offset = np.array(para_offset)
        if type(insuff_offset[0]) == dict:
            insuff_offset = [ins['insuff_offset'] for ins in insuff_offset]
            
        span_scores = outs[0][:, :, None] + outs[1][:, None]  # [bs, maxseqleninbatch, maxseqleninbatch] start_score+end_score
        max_seq_len = span_scores.size(1)
        span_mask = np.tril(np.triu(np.ones((max_seq_len, max_seq_len)), 0), max_ans_len)          # [maxseqleninbatch, maxseqleninbatch] everything before possible start and after possible start + max ans len masked 
        span_mask = span_scores.data.new(max_seq_len, max_seq_len).copy_(torch.from_numpy(span_mask))   # np->torch [maxseqleninbatch, maxseqleninbatch]
        span_scores_masked = span_scores.float().masked_fill((1 - span_mask[None].expand_as(span_scores)).bool(), -1e10).type_as(span_scores)  # [bs, maxseqleninbatch, maxseqleninbatch] with everything before start or after start + max ans len masked 
        start_position = span_scores_masked.max(dim=2)[0].max(dim=1)[1]  # [bs] .max returns values in [0], indices in [1]
        end_position = span_scores_masked.max(dim=2)[1].gather(1, start_position.unsqueeze(1)).squeeze(1) # [bs]
        answer_scores = span_scores_masked.max(dim=2)[0].max(dim=1)[0].tolist() # [bs]
        start_position_ = list(np.array(start_position.tolist()) - para_offset)  #para masking starts after base question so can predict span in query sents
        end_position_ = list(np.array(end_position.tolist()) - para_offset) 

        insuff_scores = [] # [bs]  obtain score of [unused0]/insuff evidence token - can now look at magnitude of answer_score - insuff_scores or just magnitude of insuff_score 
        ans_delta = []
        start_logits = outs[0]  # [bs, maxseqleninbatch]
        end_logits = outs[1]    # [bs, maxseqleninbatch]
        for idx, offset in enumerate(insuff_offset):
            insuff_scores.append( float( start_logits[idx, offset] + end_logits[idx, offset] ) )    # offset-2=Y, -1=N, +0=[unused0]/insuff
            ans_delta.append(answer_scores[idx] - insuff_scores[-1])
    
        self.span_scores = answer_scores
        self.start_position_ = start_position_
        self.end_position_ = end_position_
        self.insuff_scores = insuff_scores
        self.ans_delta = ans_delta
        self.span_scores = []
        self.pred_strs = []
        
        for idx in range(item_list):             # get the predicted answer strings
            start = start_position_[idx]
            end = end_position_[idx]
            tok_to_orig_index = item_list['tok_to_orig_index'][idx]
            doc_tokens = item_list['doc_tokens'][idx]
            wp_tokens = item_list['wp_tokens'][idx]
            orig_doc_start = tok_to_orig_index[start]
            orig_doc_end = tok_to_orig_index[end]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_tokens = wp_tokens[start:end+1]
            tok_text = " ".join(tok_tokens)
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            self.pred_strs.append( get_final_text(tok_text, orig_text, do_lower_case=True, verbose_logging=False).strip() )
        return
            
    def get_answer(self, idx):
        return self.pred_strs[idx]
        




