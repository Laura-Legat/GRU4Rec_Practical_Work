from gru4rec_pytorch import SessionDataIterator
import torch
import sys
sys.path.append('/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI')
from data_sampler import get_rel_int_dict
import numpy as np

# define helper function for reverse mapping indices -> item IDs, as
# they are needed in their initial form for ex2vec
def get_itemId(gru, idx):
  """
    Maps GRU4Rec-specific indices back to item IDs.

    Args:
        idx: GRU4Rec index from item ID map -> int
    Returns:
        Item ID which corresponds to the index argument -> int
  """
  return gru.data_iterator.itemidmap.index[gru.data_iterator.itemidmap.iloc[idx]]

@torch.no_grad() # disable grad computation for this function
def batch_eval(gru, test_data, cutoff=[20], batch_size=50, mode='conservative', item_key='itemId', user_key='userId', rel_int_key='relational_interval', session_key='SessionId', time_key='timestamp', combination=False, k=10, ex2vec=None):
    """
    Evaluates the model's recall and MRR for varying cutoffs.

    Args:
        gru: The GRU4Rec model being evaluated
        ex2vec: Trained Ex2Vec model for scoring gru4rec's top-k items
        test_data: Dataset to evaluate the GRU4Rec model on
        cutoff: List of values to use as cutoffs for MRR and recall calculations
        batch_size: The batch size for processing the test dataset
        mode: Ranking mode used in the evaluation (standard, conservative, median)
        item_key: Column name of itemid column
        session_key: Column name of session id column
        combination: If reranking should be done with ex2vec (=True), or normal gru4rec evaluation (=False)
        time_key: Column name of timestamp column
    """
    if gru.error_during_train: 
        raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
    
    # define and initialize structures to store the metrics for each cutoff
    recall = dict()
    mrr = dict()
    for c in cutoff:
        recall[c] = 0
        mrr[c] = 0
    
    if combination:
        # get modifiable copy of rel_int_dict
        rel_int_dict = get_rel_int_dict().copy()

    # structure for storing the hidden states
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)

    # prepare dataloader
    data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key, user_key, session_key, rel_int_key, time_key, device=gru.device, itemidmap=gru.data_iterator.itemidmap)

    n = 0
    for in_idxs, out_idxs, userids, sessionids, rel_ints in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
        for h in H: h.detach_()

        O = gru.model.forward(in_idxs, H, None, training=False) # for each item in in_idxs, calcuate a next-item probability for all items in the whole dataset (batch_size, n_all_items), e.g. (50, 879) or (10,879)

        if combination:
            top_k_scores, top_indices = torch.topk(O, k, dim=1)
            in_idx_ids = [get_itemId(gru, i) for i in in_idxs.tolist()]
            out_idx_ids = [get_itemId(gru, i) for i in out_idxs.tolist()]
            #print("\nInput item ids: ", in_idx_ids)
            #print("Out idxs: ", out_idx_ids)
            #print("From users: ", userids)
            #print("With rel ints: ", rel_ints)

            # update rel_ints dict with current, in-session repetitions
            for item, user, rel_int in zip(in_idx_ids, userids, rel_ints):
                key = (user, item)
                rel_int_dict[key] = rel_int  

            top_k_scores, top_indices = torch.topk(O, k, dim=1) # extract top k predicted next items
            top_k_item_ids = [get_itemId(gru, i) for i in top_indices.tolist()] # convert them from gru4rec indices to item ids which can be used with ex2vec
            top_k_item_ids = [top_k_item_ids[i].tolist() for i in range(len(top_k_item_ids))] # transform [Index([score,score,...]), Index([score, score]),...] to [[score, score,...], [score, score,...],...]
            #print(f"Top-{k} item recommendations for {in_idx_ids}, respectively: {top_k_item_ids}")

            # expand usersids along recommended item lists, and flatten them using sum()
            expanded_userids = sum([[userid] * len(items) for userid, items in zip(userids, top_k_item_ids)], [])
            # flatten recommended items to 1d array
            flattened_top_k_items = sum(top_k_item_ids, [])

            # for current user ids and recommended items, extract relational interval from rel_int dict
            rel_ints = [rel_int_dict.get((user, item), []) for user, item in zip(expanded_userids, flattened_top_k_items)]
            # pad relational intervals with -1 until length 50
            padded_rel_ints = []
            for rel_int in rel_ints:
                rel_int = np.pad(rel_int, (0, 50-len(rel_int)), constant_values=-1)
                padded_rel_ints.append(rel_int)

            # construct model input as tensors and move them to GPU
            user_tensor = torch.tensor(expanded_userids).cuda()
            item_tensor = torch.tensor(flattened_top_k_items).cuda()
            rel_int_tensor = torch.tensor(padded_rel_ints).cuda()

            # scoring top-k next-item predictions with ex2vec
            if ex2vec:
              scores = ex2vec(user_tensor, item_tensor, rel_int_tensor)
            # split up flattened scores into list of lists again
            scores = [scores[i:i+k] for i in range(0,len(scores),k)]
            # convert recommendation tensors to lists, so we get a list of lists, instead of a list of tensors
            scores = [score_tensor.tolist() for score_tensor in scores]
            #print(f'Ex2vec scores for top-{k}: {scores}')

            # reranking based on the score
            reranked_scores = rerank(top_k_item_ids, scores)
            #print(f'Reranked items for {in_idx_ids}, respectively: {reranked_scores}')

            combined_ranks = []
            for batch_i, out_idx in enumerate(out_idxs):
                try:
                    index = reranked_scores[batch_i].index(out_idx) # search if i-th out_idx (target item) is in the i-th sublist of the reranked score lists
                    combined_ranks.append(index + 1) # if yes, append the index it appears at + 1 to get a rank comparison with the otherwise calculated ranks
                except ValueError:
                    combined_ranks.append(len(reranked_scores[batch_i]) + 1) # if index does not appear, give a rank that is outside of the list length (=invalid rank)
            combined_ranks = torch.tensor(combined_ranks).cuda()
            #print(combined_ranks)

            for c in cutoff:
                recall[c] += (combined_ranks <= c).sum().cpu().numpy()
                mrr[c] += ((combined_ranks <= c) / combined_ranks.float()).sum().cpu().numpy()
        else:
            oscores = O.T # (879,50)
            tscores = torch.diag(oscores[out_idxs]) # select scores of the true next item (out_idx)
            #oscores = top_k_scores.T

            if mode == 'standard': ranks = (oscores > tscores).sum(dim=0) + 1 # calculates how many of the 879 item scores (for one particular in_idx) have scored higher than the corresponding true item (corresponding out_idx) + 1 for actual placement (1st place instead of 0th place)
            elif mode == 'conservative': ranks = (oscores >= tscores).sum(dim=0)
            elif mode == 'median':  ranks = (oscores > tscores).sum(dim=0) + 0.5*((oscores == tscores).dim(axis=0) - 1) + 1
            else: raise NotImplementedError
            #print(ranks)

            for c in cutoff:
                recall[c] += (ranks <= c).sum().cpu().numpy() #e.g. if cutoff==5, then it counts how often the most relevant item is among the top 5  ranks
                mrr[c] += ((ranks <= c) / ranks.float()).sum().cpu().numpy()
        n += O.shape[0] # n += batch_size

    for c in cutoff:
        recall[c] /= n # avg recall over all processed batches
        mrr[c] /= n # avg mrr over all processed batches
    return recall, mrr

def rerank(gru4rec_items, ex2vec_scores):
    """
    Reranks recommended items based on Ex2Vecs scores for those items;

    Args:
        gru4rec_items: List of lists of recommended next items, where each child list contains the top-k recommendations for the next items for one timestep
        ex2vec_scores: The corresponding scores for each itemin gru4rec_items

    Returns:
        reranked_items: List of items in re-ranked order, based on their ex2vec score -> List
    """

    # define structure for reranked item list
    reranked_items = []
    for item_list, score_list in zip(gru4rec_items, ex2vec_scores): # go through outer list
        # pair each item with its score in inner lists
        item_score_pairs = list(zip(item_list, score_list)) # [(itemid, score), (itemid, score),...]

        # sort pairs based on score value
        sorted_item_score_pairs = sorted(item_score_pairs, key=lambda x: x[1], reverse=True)

        #extract the items from the pairs
        sorted_items = [item for item, score in sorted_item_score_pairs]
        reranked_items.append(sorted_items)

    return reranked_items