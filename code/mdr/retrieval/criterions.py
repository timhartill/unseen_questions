#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Portions Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F



def mhop_loss_var(model, batch, args):
    """ Generalised loss calculation over batch of samples with varying steps required.
        
    """
    outputs = model(batch)  #TJH returns {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1} 
                            #TJH each dict element a tensor of [bs, hidden_size=hs] so eg c1 order = start paras of each question in batch
                            #TJH c1=start para, c2=bridge/2nd para, q_sp1=q + start para
                            #TJH make model return: {'q': [q, q_sp1, q_sp1_sp2, ..., q_sp1_.._spn], 'c': [[c1], [c2], .., [cn]], "neg": [[neg_1], [neg_2], ... [negn]]} 
                            #TJH build eg all_ctx = torch.cat([c for c in outputs['c'] ], dim=1)
                            #TJH Must be an equal number of paras in each sample so need to fill with negs
                            #TJH and need to pad 'q' with fake questions for samples with < max_hops..
    #loss_fct = CrossEntropyLoss(ignore_index=-100)
    bs = outputs['q'].size(0)
    act_hops = outputs['act_hops']   # [bs] Actual # steps per sample
    max_hops = outputs['c'].size(0)  # outputs["c"] should be padded to max_hops with neg paras in samples with act_hops < max_hops steps
    all_ctx = torch.cat([c for c in outputs['c']], dim=0) # [bs * max_hops, hs]
    neg_ctx = torch.cat([neg.unsqueeze(1) for neg in outputs['neg']], dim=1) # [bs, #negs, hs]
    # outputs['q'] must be padded to max_hops steps where act_hops < max_hops 
    all_q_reshaped = torch.cat([qs.unsqueeze(1) for qs in outputs['q']], dim=1) # [bs,1,hs] cat [bs,1,hs] .. = [bs, max_hops, hs]
    scores_all_hops = torch.matmul(all_q_reshaped, all_ctx.t())  # [bs, #qs, hs] matmul [hs, bs * #c] = [bs, #qs, bs * #c] ie here max_hops = #qs = #c: [bs, max_hops, bs * max_hops]

    cell_0 = torch.zeros(bs, bs)  
    cell_eye = torch.eye(bs)
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

    if args.momentum:
        n_gpu = torch.cuda.device_count() #Updated based on https://github.com/facebookresearch/multihop_dense_retrieval/pull/14/commits/96a0df6620ab02b231a1448373da7f59f615dae1
        if n_gpu > 1:
            mdl = model.module
        else:
            mdl = model
        #queue_neg_scores_1 = torch.mm(outputs["q"], mdl.queue.clone().detach().t())
        #queue_neg_scores_2 = torch.mm(outputs["q_sp1"], mdl.queue.clone().detach().t())
        queue_neg_scores_all = torch.matmul(all_q_reshaped, mdl.queue.clone().detach().t())  # [bs, #qs, hs] matmul [hs, #qnegs] = [bs, #qs, #qnegs]
        #scores_1_hop = torch.cat([scores_1_hop, queue_neg_scores_1], dim=1)
        #scores_2_hop = torch.cat([scores_2_hop, queue_neg_scores_2], dim=1)
        scores_all_hops = torch.cat([scores_all_hops, queue_neg_scores_all], dim=2) # [bs, #qs, bs*#c + #negs] cat [bs, #qs, #qnegs] = [bs, #qs, bs*#c + #negs + #qnegs]
        
        mdl.dequeue_and_enqueue(all_ctx.detach())  #Updated based on https://github.com/facebookresearch/multihop_dense_retrieval/pull/14/commits/96a0df6620ab02b231a1448373da7f59f615dae1


    ce = CrossEntropyLoss(ignore_index=-100, reduction='none')
    target_1_hop = torch.arange(bs)
    all_targets_all_hops = torch.cat([target_1_hop.unsqueeze(1) + (i*bs) for i in range(max_hops) ], dim=1) # [bs, max_hops] Target "next para" idx.
    for i in range(max_hops): # set target label to be ignored in ce if hop in target tensor is > actual hops for a sample ie ignore padding queries 
        curr_hop = i + 1
        for j in range(bs):
            if curr_hop > act_hops[j]:
                all_targets_all_hops[j, i] = -100
    retrieve_loss = ce(scores_all_hops.transpose(1,2), all_targets_all_hops)  # [bs, max_hops]
    final_loss_nonzero = torch.cat([retrieve_loss[ retrieve_loss[:,i].nonzero(), i ].mean().unsqueeze(0) for i in range(max_hops)] ).sum() # sum( mean_over_non-zero(hop_n) ) - ce sets outputs with label -100 to 0.0
    return final_loss_nonzero
    
    #import numpy as np
    #bs = 3
    #hs = 768
    #max_hops = 3  #TJH max_hops = #c ie max 3 hop questions will have 3 context paras of which the actual # hops will be the # true gold paras and the remainder negs
    #np.random.seed(42)
    #c1 = torch.tensor(np.random.randn(bs, hs), dtype=torch.float32) # 3,768  [bs, hs]
    #c2 = torch.tensor(np.random.randn(bs, hs), dtype=torch.float32) # 3,768
    #q = torch.tensor(np.random.randn(bs, hs), dtype=torch.float32) # 3,768
    #q_sp = torch.tensor(np.random.randn(bs, hs), dtype=torch.float32)  # 3,768
    
    #neg1 = torch.tensor(np.random.randn(bs, hs), dtype=torch.float32)
    #neg2 = torch.tensor(np.random.randn(bs, hs), dtype=torch.float32)

    #c3 = torch.tensor(np.random.randn(bs, hs), dtype=torch.float32) # 3,768
    #q_sp_sp2 = torch.tensor(np.random.randn(bs, hs), dtype=torch.float32)  # 3,768

    #all_ctx = torch.cat([c for c in [c1, c2, c3]], dim=0) # [6,768]  i.e [bs * #c, 768]
    
    #act_hops = torch.tensor([3,2,1], dtype=torch.int64) # [bs] = actual num of hops per sample
    
    #scores_1_hop = torch.mm(q, all_ctx.t()) #[3,768].[768,6] = [3,6] = [bs, bs*2] = [bs, bs * #c] i.e. q vs all c in batch ordered as q0_c1, q1_c1, ... q0_c2, q1_c2, ... and eventually negs are added to the end of this below
    #scores_2_hop = torch.mm(q_sp, all_ctx.t()) #[3,768].[768,6] = [3,6]

    #all_q = torch.cat([qs for qs in [q, q_sp]], dim=1)  # [3,1536]
    #all_q_reshaped = all_q.view(bs,-1, hs) # [3, 2, 768] i.e [bs, #qs, hs]
    
    #all_q_reshaped_test = torch.cat([qs.unsqueeze(1) for qs in [q, q_sp]], dim=1) # [3, 2, 768]
    #(all_q_reshaped_test == all_q_reshaped).all() # tensor(True)
    
    #all_q_reshaped = torch.cat([qs.unsqueeze(1) for qs in [q, q_sp, q_sp_sp2]], dim=1) # [bs,1,hs] cat [bs,1,hs] = [3, 2, 768]
    
    #all_ctx_reshaped = torch.cat([c1,c2], dim=1).view(bs,-1, hs).transpose(1,2) # need [bs, hs, #c]  #Note:tested that this is equivalent to torch.cat([c1.unsqueeze(1), c2.unsqueeze(1)], dim=1).transpose(1,2)
    #TJH This only multiplies each q and q_sp1 against it's own gold paras:
    #scores_all_hops = torch.bmm(all_q_reshaped, all_ctx_reshaped) # [3,2,2] ie [bs ,#qs, hs] bmm [bs, hs, #c] = [bs, #qs, #c]
    #TJH Need to multiply each q and q_sp against ALL gold paras in the batch!
    #TJH Can I multiply qs against copies of all_ctx? torch.matmul WORKS!
    #scores_all_hops = torch.matmul(all_q_reshaped, all_ctx.t())  # [bs, #qs, hs] matmul [hs, bs * #c] = [bs, #qs, bs * #c] ie [3, 2, 6] and tests confirm scores_all_hops[:,0,:] = scores_1_hop and scores_all_hops[:,1,:] = scores_2_hop


    #TODO calculate target para instead of arange
    #TODO note - don't want to -inf neg paras used to backfill smaller than max_hop context paras - replace 1s with 0s in below masking...

    
    #TODO get masking working
    #bsize = bs
    #scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize), torch.eye(bsize)], dim=1)
    #scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop)

    #TJH 1st hop only parameterized by #c (ie bsize * max_hops): INCORRECT Need hop1 = (max_hops-1)*zeros(bsize, bsize) cat (max_hops-1)*eye(bsize), hop2= (max_hops-2)*zeros(bsize, bsize) cat (maxhops-0)*eye(bsize)
    #scores_all_mask = torch.cat([torch.zeros(bsize*(max_hops-1), bsize*(max_hops-1)),torch.eye(bsize*(max_hops-1))], dim=1)
    #cell_0 = torch.zeros(bsize, bsize)  #bsize = #paras at cx
    #cell_eye = torch.eye(bsize)
    #hop_mask_list = []
    #for i in range(max_hops):
        # for max_hops = 3 :
        #hop 0 add i+1=1 cell_0 and max_hops-(i+1)=2 cell_eye
        #hop 1 add i+2=2 cell_0 and 1 cell_eye
        #hop 2 add i+3=3 cell_0 and 0 cell_eye
    #    hop_mask_list.append( torch.cat([cell_0.repeat(1,i+1), cell_eye.repeat(1,max_hops-(i+1))], dim=1) )
    #scores_all_mask = torch.cat([hm.unsqueeze(1) for hm in hop_mask_list], dim=1)  # [bs, #qs, bs*#c] [3,2,6] 

    #for j in range(bs): # dont set -inf score if curr hop in target tensor is > actual hops for a sample
    #    for i in range(max_hops):
    #            for k in range(max_hops):
    #                curr_hop = k + 1
    #                if curr_hop > act_hops[j] and i < k:
    #                    print(f"setting [{j},{i},{(k*bs)+j}] to 0. It is currently: {scores_all_mask[j, i, (k*bs)+j]}")
    #                    scores_all_mask[j, i, (k*bs)+j] = 0.0


    #scores_all_hops = scores_all_hops.float().masked_fill(scores_all_mask.bool(), float('-inf')).type_as(scores_all_hops) # [bs, #qs, bs*#c] [3,2,6]
    #(scores_1_hop==scores_all_hops[:,0]).all() # tensor(True)

    
    
    #DONE Add negs to scores_all_hops
    #neg_ctx = torch.cat([neg1.unsqueeze(1), neg2.unsqueeze(1)], dim=1) # [3, 2, 768]
    #neg_ctx_all = torch.cat([neg.unsqueeze(1) for neg in [neg1, neg2]], dim=1) # [3, 2, 768]
    #(neg_ctx_all == neg_ctx).all() # tensor(True)
    #(all_q_reshaped ==neg_ctx).all()  # tensor(False) just checking..
    #neg_ctx = torch.cat([neg.unsqueeze(1) for neg in [neg1, neg2]], dim=1) # [3, 2, 768] ie [bs, #negs, hs]
    #neg_scores_1 = torch.bmm(q.unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1) #TJH [bs,1,hs] bmm [bs,hs,#negs] = [bs,1,#negs] then squeeze(1) = [bs,#negs]
    #neg_scores_2 = torch.bmm(q_sp.unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1) #TJH [bs,1,hs] bmm [bs,hs,#negs] = [bs,1,#negs] then squeeze(1) = [bs,#negs]
    
    #neg_scores_all = torch.bmm(all_q_reshaped, neg_ctx.transpose(1,2))  # [3,2,2]: [bs, #qs, hs] bmm [bs, hs, #negs] = [bs, #qs, #negs] Confirmed neg_scores_all[:,0] = neg_scores_1 and neg_scores_all[:,1] = neg_scores_2

    #scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1) #TJH [bs, bs*2+2] ie [3,8]
    #scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1) #TJH [bs, bs*2+2] NB: Dont make start paras c1 -inf for step 2 q+c1 as you dont want q + c1 to predict c1 again 

    #scores_all_hops = torch.cat([scores_all_hops, neg_scores_all], dim=2) # TJH [3,2,8]: [bs, #qs, bs*#c] cat [bs, #qs, #negs] = [bs, #qs, bs*#c + #negs]
    #(scores_1_hop == scores_all_hops[:,0]).all() #True
    #(scores_2_hop == scores_all_hops[:,1]).all() # False since we didnt put -inf into scores_2_hop


    #DONE Get k-dimensional CE working
    #ce = CrossEntropyLoss(ignore_index=-100, reduction='none')
    #target_1_hop = torch.arange(bsize)
    #hop_1_loss = ce(scores_1_hop, target_1_hop) # tensor([72.5274, 35.9493, 48.8539])
    #hop_1_loss_agg = loss_fct(scores_1_hop, target_1_hop) # tensor(52.4436)

    #target_2_hop =  torch.arange(bsize) + bsize
    #curr_hop = 2
    #for j in range(bs):
    #    if curr_hop > act_hops[j]:
    #        target_2_hop[j] = -100
    
    #hop_2_loss = ce(scores_2_hop, target_2_hop) # tensor([22.5984, 49.3194, 43.6789])
    #hop_2_loss_agg = loss_fct(scores_2_hop, target_2_hop) # tensor(38.5322)
    
    #all_targets = torch.cat([target_1_hop.unsqueeze(1), target_2_hop.unsqueeze(1)], dim=1)
    #all_targets_all_hops = torch.cat([target_1_hop.unsqueeze(1) + (i*bsize) for i in range(max_hops) ], dim=1) # [bs, max_hops]
    
    #all_targets_all_hops_test = all_targets_all_hops.clone() # [bs, max_hops]
    #for i in range(max_hops): # set target label to be ignored in ce if curr hop in target tensor is > actual hops for a sample
    #    curr_hop = i + 1
    #    for j in range(bs):
    #        if curr_hop > act_hops[j]:
    #            all_targets_all_hops_test[j, i] = -100
            
    #retrieve_loss = ce(scores_all_hops.transpose(1,2), all_targets_all_hops)  # torch.Size([3, 2]) same as [hop_1_loss, hop_2_loss]
    #retrieve_loss_test = ce(scores_all_hops.transpose(1,2), all_targets_all_hops_test)  # torch.Size([3, 2]) same as [hop_1_loss, hop_2_loss]
    
    #retrieve_loss_agg = loss_fct(scores_all_hops.transpose(1,2), all_targets_all_hops) # tensor(45.4879)
    #hop_1_loss_agg + hop_2_loss_agg  # tensor(90.9758)  #3hop: 60.1025+36.0074 = 96.1098
    #(hop_1_loss_agg + hop_2_loss_agg) / 2 # tensor(45.4879)
    
    #final_loss = retrieve_loss.mean(dim=0).sum() # tensor(90.9758) before nega or 98.6347 after = hop_1_loss_agg + hop_2_loss_agg
    #retrieve_loss_test.mean(dim=0).sum()
    #retrieve_loss_test.mean(dim=0)
    
    #ignore_mask = retrieve_loss_test != 0
    # this works but does the torch.tensor([...]) screw up the backward pass?
    #torch.sum(torch.tensor([retrieve_loss_test[retrieve_loss_test[:,i].nonzero(), i].mean() for i in range(max_hops-1)]) ) #96.1098 matches hop_1_loss_agg + hop_2_loss_agg
    #torch.sum(torch.tensor([retrieve_loss_test[retrieve_loss_test[:,i].nonzero(), i].mean() for i in range(max_hops)]) ) #157.2357
    
    # this also works - add .unsqueeze(0) to give tensor dims then can cat them:
    #torch.cat([retrieve_loss_test[retrieve_loss_test[:,i].nonzero(), i].mean().unsqueeze(0) for i in range(max_hops-1)] ).sum() #96.1098 matches hop_1_loss_agg + hop_2_loss_agg
    #final_loss_nonzero = torch.cat([retrieve_loss_test[retrieve_loss_test[:,i].nonzero(), i].mean().unsqueeze(0) for i in range(max_hops)] ).sum() #157.2357

    
    #TJH end test #################
    
    
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
    retrieve_loss = loss_fct(scores_1_hop, target_1_hop) + loss_fct(scores_2_hop, target_2_hop)

    return retrieve_loss


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
    retrieve_loss = loss_fct(scores_1_hop, target_1_hop) + loss_fct(scores_2_hop, target_2_hop)

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
