import torch
import sys
import numpy as np
import importlib
from gru4rec_pytorch import SessionDataIterator
#sys.path.append('/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI')
from data_sampler import get_rel_int_dict, get_userId_from_mapping, get_itemId_from_mapping
import gru4rec_utils

@torch.no_grad()
def store_only(gru, test_data, batch_size=50, item_key='itemId', user_key='userId', rel_int_key='relational_interval', session_key='SessionId', time_key='timestamp', k=879, ex2vec=None, score_store_pth='./'):
    """
    Gets the scores of both GRU4Rec and Ex2Vec and stores them in a file for further analysis
    """
    # get modifiable copy of rel_int_dict
    rel_int_dict = get_rel_int_dict().copy()
    # update rel int dict once for all inetractions in test data
    for item, user, rel_int in zip(test_data['itemId'], test_data['userId'], test_data['relational_interval']):
        rel_int_dict[(user, item)] = rel_int 

    # structure for storing the hidden states
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)

    # prepare dataloader
    data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key=item_key, user_key=user_key, session_key=session_key, rel_int_key = rel_int_key, time_key=time_key, device=gru.device, itemidmap=gru.data_iterator.itemidmap) 

    # specify files to save the scores to
    gru4rec_scores_file = f'{score_store_pth}/gru4rec_scores.pt'
    ex2vec_scores_file = f'{score_store_pth}/ex2vec_scores.pt'

    with open(gru4rec_scores_file, 'wb') as gru_f, open(ex2vec_scores_file, 'wb') as ex2vec_f:
        for in_idxs, _, userids, _, rel_ints, _ in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
            for h in H: h.detach_()

            O = gru.model.forward(in_idxs, H, None, training=False) # for each item in in_idxs, calcuate a next-item probability for all items in the whole dataset (batch_size, n_all_items), e.g. (50, 879) or (10,879)
            top_k_scores, top_indices = torch.topk(O, k, dim=1) # extract top k predicted next items, (batch_size, k)
            
            # incrementally save gru4rec scores
            top_k_scores_cpu = top_k_scores.cpu()
            torch.save(top_k_scores_cpu, gru_f)
            del top_k_scores, top_k_scores_cpu
            torch.cuda.empty_cache()

            # flatten recommended items to 1d array
            flattened_top_k_items = top_indices.view(-1).cpu().numpy() #batch_size * k

            # expand usersids along recommended item lists, and flatten them using sum()
            expanded_userids = np.repeat(userids, k)

            # for current user ids and recommended items, extract relational interval from rel_int dict
            rel_ints = [rel_int_dict.get((user, item), []) for user, item in zip(expanded_userids, flattened_top_k_items)]

            # scoring top-k next-item predictions with ex2vec
            ex2vec_scores, _ = ex2vec(torch.tensor(expanded_userids).cuda(), torch.tensor(flattened_top_k_items).cuda(), torch.tensor(np.array([np.pad(rel_int, (0, 50-len(rel_int)), constant_values=-1) for rel_int in rel_ints])).cuda())
    
            # split up flattened scores into list of lists again
            ex2vec_scores = [ex2vec_scores[i:i+k] for i in range(0,len(ex2vec_scores),k)]
            ex2vec_scores = torch.stack(ex2vec_scores, dim=0) # reorder to tensor

            ex2vec_scores_cpu = ex2vec_scores.cpu()
            torch.save(ex2vec_scores_cpu, ex2vec_f)
            del ex2vec_scores, ex2vec_scores_cpu, flattened_top_k_items, expanded_userids
            torch.cuda.empty_cache()

@torch.no_grad() # disable grad computation for this function
def batch_eval(gru, test_data, cutoff=[20], batch_size=50, mode='conservative', item_key='itemId', user_key='userId', rel_int_key='relational_interval', session_key='SessionId', time_key='timestamp', combination=None, k=10, ex2vec=None, alpha_list=[0.5]):
    """
    Evaluates the model's recall and MRR for varying cutoffs.

    Args:
        gru: The GRU4Rec model being evaluated
        test_data: Dataset to evaluate the GRU4Rec model on
        cutoff: List of values to use as cutoffs for MRR and recall calculations
        batch_size: The batch size for processing the test dataset
        mode: Ranking mode used in the evaluation (standard, conservative, median)
        item_key: Column name of item ID column in the dataset
        user_key: Column name of the user ID column in the dataset
        rel_int_key: Column name of the relational interval column in the dataset
        session_key: Column name of session ID column in the dataset
        time_key: Column name of timestamp column in the dataset
        combination: How the score of ex2vec and gru4rec should be combined (str in [direct, weighted, boosted, mult]) during inference, for no combination it is None. The modes are:
            direct: Re-rank gru4rec solely based on the Ex2Vec scores
            weighted: Weighted combination of score with hyperparameter alpha, combined_score = (alpha * gru4rec_score) + ((1 - alpha) * ex2vec_score))
            boosted: Boost items where Ex2Vec has high interest scores with boosted_score = gru4rec_score + lambda * ex2vec_score
            mult: Simple ensemble model where scores are blended through multiplication, combined_score = gru4rec_score * ex2vec_score
        k: Top-k scores to choose from list of all gru4rec scores. Set k=[879] for re-ranking all items via ex2vec. k decides "re-rank all + cut" or "cut first + re-rank filtered items"
        ex2vec: Trained Ex2Vec model for scoring gru4rec's top-k items, if no combination then it is None
        alpha_list: Parameter list set for weighted/boosted combination of scores, i.e. how much to take each models' predictions into account for the final score -> List
    """
    if gru.error_during_train: 
        raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
    
    # define and initialize structures to store the metrics for each cutoff
    recall = dict()
    mrr = dict()
    
    if combination != None:
        # get modifiable copy of rel_int_dict
        rel_int_dict = get_rel_int_dict().copy()
        # update rel int dict once for all inetractions in test data
        for item, user, rel_int in zip(test_data['itemId'], test_data['userId'], test_data['relational_interval']):
            rel_int_dict[(user, item)] = rel_int 

        for c in cutoff:
            recall[c] = []
            mrr[c] = []
    else:
        for c in cutoff:
            recall[c] = 0
            mrr[c] = 0

    # structure for storing the hidden states
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)

    # prepare dataloader
    data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key=item_key, user_key=user_key, session_key=session_key, rel_int_key = rel_int_key, time_key=time_key, device=gru.device, itemidmap=gru.data_iterator.itemidmap) 
    
    n = 0
    for in_idxs, out_idxs, userids, sessionids, rel_ints, _ in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
        for h in H: h.detach_()

        O = gru.model.forward(in_idxs, H, None, training=False) # for each item in in_idxs, calcuate a next-item probability for all items in the whole dataset (batch_size, n_all_items), e.g. (50, 879) or (10,879)
        #print('GRU4Rec scores eval: ', O)

        if combination != None:
            top_k_scores, top_indices = torch.topk(O, k, dim=1) # extract top k predicted next items, (batch_size, k)

            # flatten recommended items to 1d array
            flattened_top_k_items = top_indices.view(-1).cpu().numpy() #batch_size * k

            # expand usersids along recommended item lists, and flatten them using sum()
            expanded_userids = np.repeat(userids, k)

            # for current user ids and recommended items, extract relational interval from rel_int dict
            rel_ints = [rel_int_dict.get((user, item), []) for user, item in zip(expanded_userids, flattened_top_k_items)]

            # scoring top-k next-item predictions with ex2vec
            ex2vec_scores, _ = ex2vec(torch.tensor(expanded_userids).cuda(), torch.tensor(flattened_top_k_items).cuda(), torch.tensor(np.array([np.pad(rel_int, (0, 50-len(rel_int)), constant_values=-1) for rel_int in rel_ints])).cuda())
            #print('Ex2vec scores eval: ', ex2vec_scores)
            
            # split up flattened scores into list of lists again
            ex2vec_scores = [ex2vec_scores[i:i+k] for i in range(0,len(ex2vec_scores),k)]
            ex2vec_scores = torch.stack(ex2vec_scores, dim=0) # reorder to tensor

            # reranking based on the score
            reranked_items_all_alpha = gru4rec_utils.rerank(top_indices.cuda(), top_k_scores, ex2vec_scores, alpha_list, combination) # either 879 or top-k
            alpha_ranks = [] # store list of lists of ranks per alpha
            for reranked_per_alpha in reranked_items_all_alpha:
                ranks = []
                for i, out_idx in enumerate(out_idxs): # loop through [4,5,6]
                    if out_idx in reranked_per_alpha[i]: # check if first out_idx item is in corresponding first child-tensor
                      index = (reranked_per_alpha[i] == out_idx).nonzero(as_tuple=True)[0].item() # retrieve index where target item is in subtensor of reranked items
                      ranks.append(index + 1) # if yes, append the index it appears at + 1 to get a rank comparison with the otherwise calculated ranks
                    else: # out_idx is not in sublist reranked_per_alpha[i]
                      ranks.append(len(reranked_per_alpha[i]) + 1) # if index does not appear, give a rank that is outside of the list length (=invalid rank)
                alpha_ranks.append(ranks)

            alpha_ranks = torch.tensor(np.array(alpha_ranks)).cuda()

            for c in cutoff:
                recall[c] += [sum([(rank <= c) for rank in rank_list]) for rank_list in alpha_ranks.cpu().numpy().tolist()]
                mrr[c] += [sum([((rank <= c)/rank) for rank in rank_list]) for rank_list in alpha_ranks.cpu().numpy().tolist()]
        else:
            oscores = O.T # (879,50)
            tscores = torch.diag(oscores[out_idxs]) # select scores of the true next item (out_idx)

            if mode == 'standard': ranks = (oscores > tscores).sum(dim=0) + 1 # calculates how many of the 879 item scores (for one particular in_idx) have scored higher than the corresponding true item (corresponding out_idx) + 1 for actual placement (1st place instead of 0th place)
            elif mode == 'conservative': ranks = (oscores >= tscores).sum(dim=0)
            elif mode == 'median':  ranks = (oscores > tscores).sum(dim=0) + 0.5*((oscores == tscores).dim(axis=0) - 1) + 1
            else: raise NotImplementedError

            for c in cutoff:
                recall[c] += (ranks <= c).sum().cpu().numpy() #e.g. if cutoff==5, then it counts how often the most relevant item is among the top 5  ranks
                mrr[c] += ((ranks <= c) / ranks.float()).sum().cpu().numpy()
        n += O.shape[0] # n += batch_size

    if combination != None:
        for c in cutoff:
            recall[c] = [recall[c][alpha]/n for alpha in range(len(alpha_list))]
            mrr[c] = [mrr[c][alpha]/n for alpha in range(len(alpha_list))]
    else:
        for c in cutoff:
            recall[c] /= n # avg recall over all processed batches
            mrr[c] /= n # avg mrr over all processed batches
    
    return recall, mrr