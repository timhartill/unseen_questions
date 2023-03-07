#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Portions Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


def mhop_loss_var(model, batch, args):
    """ Generalised loss calculation over batch of samples with varying steps required.
        
    """
    outstr=''
    outputs = model(batch)  # {'q': [q, q_sp1, q_sp1_sp2, ..., q_sp1_.._spn], 
                            #  'c': [sp1, sp2, .., spn], "neg": [neg1, neg2, ... negx], "act_hops":[sample1hops, ..,samplenhops], "stop_logits:[[bs, 2], [bs, 2]]"} 
                            # each dict element a max_hops len list of tensors shape [bs, hidden_size=hs] 
                            # except act_hops which is a bs len list of integer hop counts
                            # and stop_logits which is max_hops length list of [bs,2]
    bs = outputs['q'][0].size(0)
    dev = outputs['q'][0].device
    act_hops = outputs['act_hops'].squeeze(-1)   # [bs] Actual # steps per sample
    max_hops = len(outputs['c'])  # outputs["c"] should be padded to max_hops with neg paras in samples with act_hops < max_hops steps

    if args.eval_stop:
        stop_logits = torch.cat([s.unsqueeze(1) for s in outputs['stop_logits']], dim=1) # [bs, max_hops-1, 2]  q_only not included
    
    all_ctx = torch.cat([c for c in outputs['c']], dim=0) # [bs * max_hops, hs]
    neg_ctx = torch.cat([neg.unsqueeze(1) for neg in outputs['neg']], dim=1) # [bs, #negs, hs]
    # outputs['q'] must be padded to max_hops steps if act_hops < max_hops 
    all_q_reshaped = torch.cat([qs.unsqueeze(1) for qs in outputs['q']], dim=1) # [bs,1,hs] cat [bs,1,hs] .. = [bs, max_hops, hs]
    scores_all_hops = torch.matmul(all_q_reshaped, all_ctx.t())  # [bs, #qs, hs] matmul [hs, bs * #c] = [bs, #qs, bs * #c] ie here max_hops = #qs = #c: [bs, max_hops, bs * max_hops]

    if args.debug:
        outstr=f"## nans after matmul: scores_all_hops:{scores_all_hops.isnan().any()} "
        outstr+=f"bs:{bs} max_hops:{max_hops} act_hops:{act_hops.shape} all_ctx: {all_ctx.shape} neg_ctx: {neg_ctx.shape} all_q_reshaped:{all_q_reshaped.shape} scores_all_hops:{scores_all_hops.shape} " 
        outstr+=f"dtypes: act_hops:{act_hops.dtype} all_ctx: {all_ctx.dtype} neg_ctx: {neg_ctx.dtype} all_q_reshaped:{all_q_reshaped.dtype} scores_all_hops:{scores_all_hops.dtype} "

    cell_0 = torch.zeros(bs, bs).to(dev)  
    cell_eye = torch.eye(bs).to(dev)
    hop_mask_list = []
    for i in range(max_hops): # each step concat "x" cell_0's and "y" cell_eyes. "1"s will mask supporting paras so they aren't treated as neg paras 
        hop_mask_list.append( torch.cat([cell_0.repeat(1,i+1), cell_eye.repeat(1,max_hops-(i+1))], dim=1) )
    scores_all_mask = torch.cat([hm.unsqueeze(1) for hm in hop_mask_list], dim=1)  # [bs, max_hops, bs*max_hops] 
    for j in range(bs): # undo "1" for hops > actual hops for a sample
        for i in range(max_hops):
                for k in range(max_hops):
                    curr_hop = k + 1
                    if curr_hop > act_hops[j] and i < k:  #i<k since above hop_mask_list creation already set these correctly
                        #print(f"setting [{j},{i},{(k*bs)+j}] to 0. It is currently: {scores_all_mask[j, i, (k*bs)+j]}")
                        scores_all_mask[j, i, (k*bs)+j] = 0.0
    scores_all_hops = scores_all_hops.float().masked_fill(scores_all_mask.bool(), float('-inf')).type_as(scores_all_hops) # [bs, #qs, bs*#c] #qs = #c = max_hops

    if args.debug:
        outstr+=f"### after mask: scores_all_hops:{scores_all_hops.shape} {scores_all_hops.dtype} scores_all_mask: {scores_all_mask.shape} {scores_all_mask.dtype} "
        outstr+=f"dtypes: cell_0:{cell_0.dtype} cell_eye: {cell_eye.dtype} "
        outstr+=f"nans bef neg cat: scores_all_hops:{scores_all_hops.isnan().any()} scores_all_mask:{scores_all_mask.isnan().any()} "

    neg_scores_all = torch.bmm(all_q_reshaped, neg_ctx.transpose(1,2))  # [bs, #qs, hs] bmm [bs, hs, #negs] = [bs, #qs, #negs]
    scores_all_hops = torch.cat([scores_all_hops, neg_scores_all], dim=2) # [bs, #qs, bs*#c] cat [bs, #qs, #negs] = [bs, #qs, bs*#c + #negs]

    if args.debug:
        outstr+=f"####nans after neg cat: scores_all_hops:{scores_all_hops.isnan().any()} {scores_all_hops.dtype} neg_scores_all:{neg_scores_all.isnan().any()} "


    if args.momentum:
        n_gpu = torch.cuda.device_count() #Updated based on https://github.com/facebookresearch/multihop_dense_retrieval/pull/14/commits/96a0df6620ab02b231a1448373da7f59f615dae1
        if n_gpu > 1:
            mdl = model.module
        else:
            mdl = model
        queue_neg_scores_all = torch.matmul(all_q_reshaped, mdl.queue.clone().detach().t())  # [bs, #qs, hs] matmul [hs, #qnegs] = [bs, #qs, #qnegs]
        scores_all_hops = torch.cat([scores_all_hops, queue_neg_scores_all], dim=2) # [bs, #qs, bs*#c + #negs] cat [bs, #qs, #qnegs] = [bs, #qs, bs*#c + #negs + #qnegs]
        
        mdl.dequeue_and_enqueue(all_ctx.detach())  #Updated based on https://github.com/facebookresearch/multihop_dense_retrieval/pull/14/commits/96a0df6620ab02b231a1448373da7f59f615dae1

    if args.debug:
        outstr+=f"#####nans after mom: scores_all_hops:{scores_all_hops.isnan().any()} {scores_all_hops.dtype} shape:{scores_all_hops.shape}"


    if args.eval_stop:  # learn to flag stop after act_hops eg if q+sp1 and act_hops = 2 then stop but if act_hops = 3 then don't stop
        stop_ce = CrossEntropyLoss(ignore_index=-100, reduction=args.reduction if args.reduction != 'none' else 'mean')
        hop_target_idxs = torch.zeros(bs, max_hops-1, dtype=torch.int64).to(dev)
        for i in range(max_hops-1):
            for j in range(bs):
                if i > act_hops[j]-1:
                    hop_target_idxs[j, i] = -100
                elif i == act_hops[j]-1:
                    hop_target_idxs[j, i] = 1 # train stop_logits[:,:,1] to be "yes stop now" and stop_logits[:,:,0] to be "keep going"
        stop_loss = stop_ce(stop_logits.transpose(1,2), hop_target_idxs) # [bs, max_hops-1] if reduction was 'none' from ce([bs, classes=2, max_hops], [bs, target class idx={0/1}])
        if args.debug:
            outstr+=f"nans stop_loss:{stop_loss.isnan().any()} {stop_loss.dtype} {stop_loss.shape} VALUE:{stop_loss}" 
    
    target_1_hop = torch.arange(bs).to(dev)
    all_targets_all_hops = torch.cat([target_1_hop.unsqueeze(1) + (i*bs) for i in range(max_hops) ], dim=1) # [bs, max_hops] Target "next para" idx.
    for i in range(max_hops): # set target label to be ignored in ce if hop in target tensor is > actual hops for a sample ie ignore padding queries 
        curr_hop = i + 1
        for j in range(bs):
            if curr_hop > act_hops[j]:
                all_targets_all_hops[j, i] = -100
    if args.reduction == 'none':
        ce = CrossEntropyLoss(ignore_index=-100, reduction='none')
        retrieve_loss = ce(scores_all_hops.transpose(1,2), all_targets_all_hops)  # [bs, max_hops] from ce([bs, classes={bs*#c + #negs}, max_hops], [bs, target class idx={max_hops}])
        if args.debug:
            outstr+=f"nans retrieve_loss:{retrieve_loss.isnan().any()} {retrieve_loss.dtype} {retrieve_loss.shape}"
    
        include_mask = all_targets_all_hops != -100  # [bs, max_hops]
        any_not_ignore = torch.cat([include_mask[:,i].any().unsqueeze(0) for i in range(max_hops)])  # [max_hops]. Ignore columns where all act_hops < max_hops
        final_loss_nonzero = torch.cat([retrieve_loss[:,i][ include_mask[:,i] ].mean().unsqueeze(0) for i in range(max_hops) if any_not_ignore[i]] ).sum() # sum( mean_over_non-zero(hop_n) ) - ce sets outputs with label -100 to 0.0
    
        #final_loss_nonzero = torch.cat([retrieve_loss[ retrieve_loss[:,i].nonzero(), i ].mean().unsqueeze(0) for i in range(max_hops)] ).sum() # sum( mean_over_non-zero(hop_n) ) - ce sets outputs with label -100 to 0.0
        if args.debug:
            outstr+=f"nans final_loss_nonzero:{final_loss_nonzero.isnan().any()} {final_loss_nonzero.dtype} VALUE:{final_loss_nonzero}"
            #if final_loss_nonzero.isnan().any():
            #    print(retrieve_loss)
        if args.eval_stop:
            return stop_loss + args.retrieve_loss_multiplier*final_loss_nonzero, outstr  #tensor(finalnum)
        return args.retrieve_loss_multiplier*final_loss_nonzero, outstr  #tensor(finalnum)
    else:
        ce = CrossEntropyLoss(ignore_index=-100, reduction=args.reduction) #"sum"
        retrieve_loss = ce(scores_all_hops.transpose(1,2), all_targets_all_hops)  # tensor(finalnum)
        if args.debug:
            outstr+=f"nans retrieve_loss:{retrieve_loss.isnan().any()} {retrieve_loss.dtype} {retrieve_loss.shape} VALUE:{retrieve_loss}"
        if args.eval_stop:
            return stop_loss + args.retrieve_loss_multiplier*retrieve_loss, outstr
        return args.retrieve_loss_multiplier*retrieve_loss, outstr
        
    

def mhop_eval_var(outputs, args):
    
    bs = outputs['q'][0].size(0)
    dev = outputs['q'][0].device
    act_hops = outputs['act_hops'].squeeze(-1)   # [bs] Actual # steps per sample
    max_hops = len(outputs['c'])  # outputs["c"] should be padded to max_hops with neg paras in samples with act_hops < max_hops steps

    if args.eval_stop:
        stop_logits = torch.cat([s.unsqueeze(1) for s in outputs['stop_logits']], dim=1) # [bs, max_hops, 2]
    
    all_ctx = torch.cat([c for c in outputs['c']], dim=0) # [bs * max_hops, hs]
    neg_ctx = torch.cat([neg.unsqueeze(1) for neg in outputs['neg']], dim=1) # [bs, #negs, hs]

    all_q_reshaped = torch.cat([qs.unsqueeze(1) for qs in outputs['q']], dim=1) # [bs,1,hs] cat [bs,1,hs] .. = [bs, max_hops, hs]
    scores_all_hops = torch.matmul(all_q_reshaped, all_ctx.t())  # [bs, #qs, hs] matmul [hs, bs * #c] = [bs, #qs, bs * #c] ie here max_hops = #qs = #c: [bs, max_hops, bs * max_hops]

   
    cell_0 = torch.zeros(bs, bs).to(dev)  
    cell_eye = torch.eye(bs).to(dev)
    hop_mask_list = []
    for i in range(max_hops): # each step concat "x" cell_0's and "y" cell_eyes. "1"s will mask supporting paras so they aren't treated as neg paras 
        hop_mask_list.append( torch.cat([cell_0.repeat(1,i+1), cell_eye.repeat(1,max_hops-(i+1))], dim=1) )
    scores_all_mask = torch.cat([hm.unsqueeze(1) for hm in hop_mask_list], dim=1)  # [bs, max_hops, bs*max_hops] 
    for j in range(bs): # undo "1" for hops > actual hops for a sample
        for i in range(max_hops):
                for k in range(max_hops):
                    curr_hop = k + 1
                    if curr_hop > act_hops[j] and i < k:  #i<k since above hop_mask_list creation already set these correctly
                        #print(f"setting [{j},{i},{(k*bs)+j}] to 0. It is currently: {scores_all_mask[j, i, (k*bs)+j]}")
                        scores_all_mask[j, i, (k*bs)+j] = 0.0
    scores_all_hops = scores_all_hops.float().masked_fill(scores_all_mask.bool(), float('-inf')).type_as(scores_all_hops) # [bs, #qs, bs*#c] #qs = #c = max_hops

    neg_scores_all = torch.bmm(all_q_reshaped, neg_ctx.transpose(1,2))  # [bs, #qs, hs] bmm [bs, hs, #negs] = [bs, #qs, #negs]
    scores_all_hops = torch.cat([scores_all_hops, neg_scores_all], dim=2) # [bs, #qs, bs*#c] cat [bs, #qs, #negs] = [bs, #qs, bs*#c + #negs]

    if args.eval_stop: # stop on hop accuracy
        stop_pred = stop_logits.argmax(dim=2)  # [bs, max_hops-1]
        hop_target_idxs = torch.zeros(bs, max_hops-1, dtype=torch.int64).to(dev)
        for i in range(max_hops-1):
            for j in range(bs):
                if i == act_hops[j]-1:
                    hop_target_idxs[j, i] = 1
    
        stop_acc = (stop_pred == hop_target_idxs).float().cpu().numpy()  # [bs, max_hops-1] with 1.0 / 0.0 in cells
        for i in range(max_hops-1):
            for j in range(bs):
                if i > act_hops[j]-1:
                    stop_acc[j, i] = 0.0  # ignore acc where current hop is greater than actual # of hops in this sample
        correct_counts = stop_acc.sum(axis=1)  # [bs]
        act_hops_denom = np.array([ah.item() if ah < max_hops else max_hops-1 for ah in act_hops]) # [bs] if sample has max_hops hops dont add 1 since we don't process the final q+sp1+...+spmax_hops    
        accuracies_per_sample = correct_counts / act_hops_denom # [bs] accuracy by sample - could be used to determine accuracy by sample type eg all 1 hop samples
        accuracies_per_hop = {}  # accuracies by hop # like rrs
        for i in range(max_hops-1):
            accuracies_per_hop[i+2] = []
            for j in range(bs):
                if i <= act_hops[j]-1:
                    accuracies_per_hop[i+2].append( int(stop_acc[j, i]) ) # Nb: accuracy of all eg 1 hop stop preds not the accuracy of only samples from a 1 hop dataset
    
    target_1_hop = torch.arange(bs).to(dev)
    all_targets_all_hops = torch.cat([target_1_hop.unsqueeze(1) + (i*bs) for i in range(max_hops) ], dim=1) # [bs, max_hops] Target "next para" idx.

    ranked_all_hops = scores_all_hops.argsort(dim=2, descending=True)  #[bs, #qs, bs*#c + #negs]
    idx_2ranked_all = ranked_all_hops.argsort(dim=2)                   #[bs, #qs, bs*#c + #negs]
    rrs = {}
    for curr_hop in range(max_hops):
        rrs[curr_hop+1] = []
        for t, idx2ranked in zip(all_targets_all_hops[:,curr_hop], idx_2ranked_all[:,curr_hop,:]): 
            if curr_hop+1 <= act_hops[t % bs]: # ignore ranking where curr_hop is greater than the actual # of hops in this sample
                #print(f"hop: {curr_hop+1}: {t}: {idx2ranked[t].item() + 1}")    #Matches hop1/hop2 below..
                rrs[curr_hop+1].append( 1 / (idx2ranked[t].item() + 1) )
    
    if args.eval_stop:
        return rrs, accuracies_per_hop, accuracies_per_sample.tolist()
    return rrs



def mhop_loss(model, batch, args):
    """ Original MDR mhop_loss """
    outputs = model(batch)  #TJH returns {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1} 
                            #TJH each dict element a tensor of [bs, hidden_size=hs] so eg c1 order = start paras of each question in batch
                            #TJH c1=start para, c2=bridge/2nd para, q_sp1=q + start para
                            #TJH make model return: {'q': [q, q_sp1, q_sp1_sp2, ..., q_sp1_.._spn], 'c': [[c1], [c2], .., [cn]], "neg": [[neg_1], [neg_2], ... [negn]]} 
                            #TJH build eg all_ctx = torch.cat([c for c in outputs['c'] ], dim=1)
                            #TJH Must be an equal number of paras in each sample so need to fill with negs
                            #TJH and need to pad 'q' with fake questions for samples with < max_hops..
    loss_fct = CrossEntropyLoss(ignore_index=-100)    
    
    all_ctx = torch.cat([outputs['c1'], outputs['c2']], dim=0)  #TJH output [bs*2, hs]
    #TJH unsqueeze(1) makes shape [bs, 1, hiddensize] so neg_ctx shape = [bs, 2, hs]
    neg_ctx = torch.cat([outputs["neg_1"].unsqueeze(1), outputs["neg_2"].unsqueeze(1)], dim=1) # B x 2 x M x h  #TJH: [bs, #negs, hs]
    
    scores_1_hop = torch.mm(outputs["q"], all_ctx.t()) #TJH [bs, hs] . [hs, bs*2] = [bs, bs*2] = [bs, all c1s then all c2s]
    neg_scores_1 = torch.bmm(outputs["q"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1) #TJH [bs,1,hs] bmm [bs,hs,2] = [bs,1,2] then squeeze(1) = [bs,2]
    scores_2_hop = torch.mm(outputs["q_sp1"], all_ctx.t())  #TJH [bs, bs*2]
    neg_scores_2 = torch.bmm(outputs["q_sp1"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1) #TJH [bs,2]

    # mask the 1st hop
    bsize = outputs["q"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(outputs["q"].device) #TJH [bs,bs*2] the 2nd half is identity matrix - per issue 20:  the reason behind this was to avoid labeling the 2-hop supporting passage as negatives. Sometimes, the hop order might not be obvious and this is especially true for comparison questions. This gave some improvements on some initial experiments.
    scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop) #TJH put -inf in score_1_hop where there are 1's in the mask - since these paras are never labelled as targets, -inf guarantees they will never erroneously be the prediction
    scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1) #TJH [bs, bs*2+2]
    scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1) #TJH [bs, bs*2+2] NB: Dont make start paras c1 -inf for step 2 q+c1 as you dont want q + c1 to predict c1 again 

    if args.momentum:
        n_gpu = torch.cuda.device_count() #TJH Updated based on https://github.com/facebookresearch/multihop_dense_retrieval/pull/14/commits/96a0df6620ab02b231a1448373da7f59f615dae1
        if n_gpu > 1:
            # Using DataParallel, so need to call model.modue
            mdl = model.module
        else:
            mdl = model

        queue_neg_scores_1 = torch.mm(outputs["q"], mdl.queue.clone().detach().t())
        queue_neg_scores_2 = torch.mm(outputs["q_sp1"], mdl.queue.clone().detach().t())
        #queue_neg_scores_1 = torch.mm(outputs["q"], model.module.queue.clone().detach().t())
        #queue_neg_scores_2 = torch.mm(outputs["q_sp1"], model.module.queue.clone().detach().t())

        # queue_neg_scores_1 = queue_neg_scores_1 / args.temperature
        # queue_neg_scores_2 = queue_neg_scores_2 / args.temperature  

        scores_1_hop = torch.cat([scores_1_hop, queue_neg_scores_1], dim=1)
        scores_2_hop = torch.cat([scores_2_hop, queue_neg_scores_2], dim=1)
        
        #model.module.dequeue_and_enqueue(all_ctx.detach())
        mdl.dequeue_and_enqueue(all_ctx.detach())  #TJH Updated based on https://github.com/facebookresearch/multihop_dense_retrieval/pull/14/commits/96a0df6620ab02b231a1448373da7f59f615dae1
        # model.module.momentum_update_key_encoder()

    #TJH correct next para encoding from q0 is at scores_1_hop[0, 0] and for q1 is at [1,1] etc: 
    target_1_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) 
    #TJH correct next para encoding from q0+sp is at scores_1_hop[0, bs] and for q1+sp is at [1,bs+1] etc: 
    target_2_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) + outputs["q"].size(0) 

    #TJH loss_fct(scores_1_hop, target_1_hop) = loss_fct([bs, bs*2+2], [bs] elements containing correct "class"/idx into bs*2+2)
    #TJH CE Input:  (bs, c) where C = number of classes. Look at K-dim CE loss!
    retrieve_loss = loss_fct(scores_1_hop, target_1_hop) + loss_fct(scores_2_hop, target_2_hop) #tensor(num1) + tensor(num2) = tensor(num3)

    return retrieve_loss




def mhop_eval(outputs, args):
    all_ctx = torch.cat([outputs['c1'], outputs['c2']], dim=0)
    neg_ctx = torch.cat([outputs["neg_1"].unsqueeze(1), outputs["neg_2"].unsqueeze(1)], dim=1)


    scores_1_hop = torch.mm(outputs["q"], all_ctx.t())
    neg_scores_1 = torch.bmm(outputs["q"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)
    scores_2_hop = torch.mm(outputs["q_sp1"], all_ctx.t())
    neg_scores_2 = torch.bmm(outputs["q_sp1"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)


    bsize = outputs["q"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(outputs["q"].device)
    scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop)
    scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1)
    scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1)
    target_1_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device)
    target_2_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) + outputs["q"].size(0)

    ranked_1_hop = scores_1_hop.argsort(dim=1, descending=True)
    ranked_2_hop = scores_2_hop.argsort(dim=1, descending=True)
    idx2ranked_1 = ranked_1_hop.argsort(dim=1)
    idx2ranked_2 = ranked_2_hop.argsort(dim=1)
    rrs_1, rrs_2 = [], []
    for t, idx2ranked in zip(target_1_hop, idx2ranked_1):
        rrs_1.append(1 / (idx2ranked[t].item() + 1))
    for t, idx2ranked in zip(target_2_hop, idx2ranked_2):
        rrs_2.append(1 / (idx2ranked[t].item() + 1))
    
    return {"rrs_1": rrs_1, "rrs_2": rrs_2}


def unified_loss(model, batch, args):

    outputs = model(batch)
    all_ctx = torch.cat([outputs['c1'], outputs['c2']], dim=0)
    neg_ctx = torch.cat([outputs["neg_1"].unsqueeze(1), outputs["neg_2"].unsqueeze(1)], dim=1)
    scores_1_hop = torch.mm(outputs["q"], all_ctx.t())
    neg_scores_1 = torch.bmm(outputs["q"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)
    scores_2_hop = torch.mm(outputs["q_sp1"], all_ctx.t())
    neg_scores_2 = torch.bmm(outputs["q_sp1"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)

    # mask for 1st hop
    bsize = outputs["q"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(outputs["q"].device)
    scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop)
    scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1)
    scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1)

    # TJH: "stop_targets" is collated over list of "stop", "stop" is [0] or [1] => stop_targets = [bs, 1] with elements 0 or 1 for not stop/stop.. 
    # TJH abc.view(-1) takes [bs,1] -> [bs]
    # TJH stop_logits = [bs, 2]
    stop_loss = F.cross_entropy(outputs["stop_logits"], batch["stop_targets"].view(-1), reduction="sum")  

    target_1_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device)
    target_2_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) + outputs["q"].size(0)


    retrieve_loss = F.cross_entropy(scores_1_hop, target_1_hop, reduction="sum") + (F.cross_entropy(scores_2_hop, target_2_hop, reduction="none") * batch["stop_targets"].view(-1)).sum()

    return retrieve_loss + stop_loss

def unified_eval(outputs, batch):
    all_ctx = torch.cat([outputs['c1'], outputs['c2']], dim=0)
    neg_ctx = torch.cat([outputs["neg_1"].unsqueeze(1), outputs["neg_2"].unsqueeze(1)], dim=1)
    scores_1_hop = torch.mm(outputs["q"], all_ctx.t())
    scores_2_hop = torch.mm(outputs["q_sp1"], all_ctx.t())
    neg_scores_1 = torch.bmm(outputs["q"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)
    neg_scores_2 = torch.bmm(outputs["q_sp1"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)
    bsize = outputs["q"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(outputs["q"].device)
    scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop)
    scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1)
    scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1)
    target_1_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device)
    target_2_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) + outputs["q"].size(0)

    # stop accuracy
    stop_pred = outputs["stop_logits"].argmax(dim=1)
    stop_targets = batch["stop_targets"].view(-1)
    stop_acc = (stop_pred == stop_targets).float().tolist()

    ranked_1_hop = scores_1_hop.argsort(dim=1, descending=True)
    ranked_2_hop = scores_2_hop.argsort(dim=1, descending=True)
    idx2ranked_1 = ranked_1_hop.argsort(dim=1)
    idx2ranked_2 = ranked_2_hop.argsort(dim=1)

    rrs_1_mhop, rrs_2_mhop, rrs_nq = [], [], []
    for t1, idx2ranked1, t2, idx2ranked2, stop in zip(target_1_hop, idx2ranked_1, target_2_hop, idx2ranked_2, stop_targets):
        if stop: # 
            rrs_1_mhop.append(1 / (idx2ranked1[t1].item() + 1))
            rrs_2_mhop.append(1 / (idx2ranked2[t2].item() + 1))
        else:
            rrs_nq.append(1 / (idx2ranked1[t1].item() + 1))

    return {
        "stop_acc": stop_acc, 
        "rrs_1_mhop": rrs_1_mhop,
        "rrs_2_mhop": rrs_2_mhop,
        "rrs_nq": rrs_nq
        }
