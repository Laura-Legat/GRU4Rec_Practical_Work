import h5py
import torch
from tqdm import tqdm
import numpy as np
from gru4rec_pytorch import SessionDataIterator
#sys.path.append('/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI')
from data_sampler import get_rel_int_dict, get_userId_from_mapping, get_itemId_from_mapping
import gru4rec_utils

@torch.no_grad()
def store_only(gru, test_data, batch_size=4096, item_key='itemId', user_key='userId', rel_int_key='relational_interval', session_key='SessionId', time_key='timestamp', k=100, ex2vec=None, score_store_pth='.'):
    """
    Gets the scores of both GRU4Rec and Ex2Vec and stores them in a file along with the corresponding items and out_idx for further analysis and metric calculation.

    Args:
        gru: The GRU4Rec model being evaluated
        test_data: Dataset to evaluate the GRU4Rec model on
        batch_size: The batch size for processing the test dataset
        item_key: Column name of item ID column in the dataset
        user_key: Column name of the user ID column in the dataset
        rel_int_key: Column name of the relational interval column in the dataset
        session_key: Column name of session ID column in the dataset
        time_key: Column name of timestamp column in the dataset
        k: Top-k scores to choose from list of all gru4rec scores. Set k=[879] for re-ranking all items via ex2vec. k decides "re-rank all + cut" or "cut first + re-rank filtered items"
        ex2vec: Trained Ex2Vec model for scoring gru4rec's top-k items
        score_store_pth: The folder where to store the .h5 file containing the scores
    """        

    # get modifiable copy of rel_int_dict
    rel_int_dict = get_rel_int_dict().copy()
    # update rel int dict once for all inetractions in test data
    for item, user, rel_int in zip(test_data['itemId'], test_data['userId'], test_data['relational_interval']):
        rel_int_dict[(user, item)] = rel_int 

    total_batches = len(test_data) // batch_size
    if len(test_data) % batch_size != 0:
        total_batches += 1

    # structure for storing the hidden states
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)

    # prepare dataloader
    data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key=item_key, user_key=user_key, session_key=session_key, rel_int_key = rel_int_key, time_key=time_key, device=gru.device, itemidmap=gru.data_iterator.itemidmap) 

    with h5py.File(score_store_pth + '/model_scores.h5', 'a') as h5_f: # iteratively write to h5 file for speed-improvement

        # generate separate datasets to store the top-k gru4rec scores, the corresponding top-k next-item predicted items, the ex2vec scores and the targets
        gru4rec_ds = h5_f.create_dataset('gru4rec_scores', shape=(0, batch_size, k), maxshape=(None, batch_size, k), chunks=(1, batch_size, k), fillvalue=-7, compression='gzip', compression_opts=6)
        gru4rec_items_ds = h5_f.create_dataset('gru4rec_items', shape=(0, batch_size, k), maxshape=(None, batch_size, k), chunks=(1, batch_size, k), fillvalue=-7, compression='gzip', compression_opts=6)
        ex2vec_ds = h5_f.create_dataset('ex2vec_scores', shape=(0, batch_size, k), maxshape=(None, batch_size, k), chunks=(1, batch_size, k), fillvalue=-7, compression='gzip', compression_opts=6)
        targets_ds = h5_f.create_dataset('targets', shape=(0, batch_size), maxshape=(None, batch_size), chunks=(1, batch_size,), fillvalue=-7, compression='gzip', compression_opts=6)

        with tqdm(total=total_batches, desc=f'Processing batch', unit='batch', ncols=100) as pbar:
            for batch_idx, (in_idxs, out_idxs, userids, _, rel_ints, _) in enumerate(data_iterator(enable_neg_samples=False, reset_hook=reset_hook)):
                for h in H: h.detach_()
                
                O = gru.model.forward(in_idxs, H, None, training=False) # for each item in in_idxs, calcuate a next-item probability for all items in the whole dataset (batch_size, n_all_items), e.g. (50, 879)
                
                top_k_scores, top_indices = torch.topk(O, k, dim=1) # extract top k (=100) predicted next items, (batch_size, k), e.g. (50, 100)
                
                # flatten recommended items to 1d array
                flattened_top_k_items = top_indices.view(-1) #batch_size * k, e.g. 50*100 = (5000,)
                
                # expand usersids along recommended item lists, and flatten them using sum()
                expanded_userids = np.repeat(userids, k) # batch_size*k, e.g. (5000,)
                
                # for current user ids and recommended items, extract relational interval from rel_int dict
                rel_ints = [rel_int_dict.get((user, item), []) for user, item in zip(expanded_userids, flattened_top_k_items)]
                
                # scoring top-k next-item predictions with ex2vec, we get (batch_size*k,), e.g. (5000,)
                ex2vec_scores, _ = ex2vec(torch.tensor(expanded_userids).cuda(), torch.tensor(flattened_top_k_items).cuda(), torch.tensor(np.array([np.pad(rel_int, (0, 50-len(rel_int)), constant_values=-1) for rel_int in rel_ints])).cuda())
                
                # split up flattened scores into list of lists again, [50,100]
                ex2vec_scores = [ex2vec_scores[i:i+k] for i in range(0,len(ex2vec_scores),k)]
                ex2vec_scores = torch.stack(ex2vec_scores, dim=0) # reorder to tensor and store

                # resize hdf5 file accordingly and store batch
                top_k_scores = np.expand_dims(top_k_scores.cpu().numpy(), axis=0)
                gru4rec_ds.resize((batch_idx + 1, batch_size, k))
                gru4rec_ds[batch_idx, :top_k_scores.shape[1], :] = top_k_scores

                # resize and fill gru4rec item dataset
                top_indices = np.expand_dims(top_indices.cpu().numpy(), axis=0)
                gru4rec_items_ds.resize((batch_idx+1, batch_size, k))
                gru4rec_items_ds[batch_idx, :top_indices.shape[1], :] = top_indices
                
                # resize and fill ex2vec ds
                ex2vec_scores = np.expand_dims(ex2vec_scores.cpu().numpy(), axis=0)
                ex2vec_ds.resize((batch_idx+1, batch_size, k))
                ex2vec_ds[batch_idx, :ex2vec_scores.shape[1], :] = ex2vec_scores

                # resize and fill targets
                target_idxs = out_idxs.cpu().numpy()
                targets_ds.resize((batch_idx+1, batch_size))
                targets_ds[batch_idx, :target_idxs.shape[0]] = target_idxs
                
                # Clean up memory
                del top_k_scores, top_indices, flattened_top_k_items, expanded_userids, ex2vec_scores, target_idxs
                
                pbar.update(1) # update progress bar

def calc_metrics_from_scores(eval_ds_path, alpha, combination_mode, eval_mode='standard', cutoffs=[1,5,10,20]):
    """
    This function computes the GRU4Rec metrics such as MRR and Recall from a HDF5 file. For that, it combines the scores, re-ranks them and calculates the metrics. 
    
    Args:
        eval_ds_path: The path whercd prac  e the .h5 file is stored
        alpha: Parameter for weighted/boosted combination of scores, i.e. how much to take each models' predictions into account for the final score -> List
        combination_mode: How the score of ex2vec and gru4rec should be combined (str in [direct, weighted, boosted, mult]) during inference, for no combination it is None. The modes are:
            direct: Re-rank gru4rec solely based on the Ex2Vec scores
            weighted: Weighted combination of score with hyperparameter alpha, combined_score = (alpha * gru4rec_score) + ((1 - alpha) * ex2vec_score))
            boosted: Boost items where Ex2Vec has high interest scores with boosted_score = gru4rec_score + lambda * ex2vec_score
            mult: Simple ensemble model where scores are blended through multiplication, combined_score = gru4rec_score * ex2vec_score
    """
    # define and initialize structures to store the metrics for each cutoff
    recall = dict()
    mrr = dict()
    recall_combined = dict()
    mrr_combined = dict()

    for c in cutoffs:
        recall[c] = 0
        mrr[c] = 0
        recall_combined[c] = 0
        mrr_combined[c] = 0

    with h5py.File(eval_ds_path, 'r') as h5_f:
        gru4rec_scores_ds = h5_f['gru4rec_scores']
        gru4rec_items_ds = h5_f['gru4rec_items']
        ex2vec_scores_ds = h5_f['ex2vec_scores']
        out_idxs_ds = h5_f['targets'] 

        n = 0
        total_batches = len(gru4rec_scores_ds)
        with tqdm(total=total_batches, desc=f'Processing batch', unit='batch', ncols=100) as pbar:
            for gru4rec_scores_chunk, gru4rec_items_chunk, ex2vec_scores_chunk, out_idx_chunks in zip(gru4rec_scores_ds.iter_chunks(), gru4rec_items_ds.iter_chunks(), ex2vec_scores_ds.iter_chunks(), out_idxs_ds.iter_chunks()):
                # convert chunks to numpy arrays
                gru4rec_scores = gru4rec_scores_ds[gru4rec_scores_chunk] # (batch_idx, batch_size, k)
                gru4rec_items = gru4rec_items_ds[gru4rec_items_chunk] # (batch_idx, batch_size, k)
                ex2vec_scores = ex2vec_scores_ds[ex2vec_scores_chunk] # (batch_idx, batch_size, k)
                out_idxs = out_idxs_ds[out_idx_chunks] # (batch_idx, batch_size,)

                #print("Orig GRU4Rec Scores: ", gru4rec_scores[:,45:55, :5])
                #print("Orig GRU4Rec Items: ", gru4rec_items[:,45:55, :5])
                #print("Orig Ex2Vec Scores: ", ex2vec_scores[:,45:55, :5])
                #print("Orig out_idxs: ", out_idxs[:,:5])

                # reshape arrays to concat batch_idx and batch_size dims 
                k = gru4rec_scores.shape[-1]
                gru4rec_scores = gru4rec_scores.reshape(-1, k) # (batch_idx*batch_size, k)
                gru4rec_items = gru4rec_items.reshape(-1, k) # (batch_idx*batch_size, k)
                ex2vec_scores = ex2vec_scores.reshape(-1, k) # (batch_idx*batch_size, k)
                out_idxs = out_idxs.reshape(-1) # (batch_idx*batch_size)

                #print("Concat GRU4Rec Scores: ", gru4rec_scores[45:55, :5])
                #print("Concat GRU4Rec Items: ", gru4rec_items[45:55, :5])
                #print("Concat Ex2Vec Scores: ", ex2vec_scores[45:55, :5])
                #print("Concat out_idxs: ", out_idxs[45:55])

                # remove fillvalue -7 (appended to datasets from store_only function)
                fillval_mask = torch.tensor((gru4rec_scores != -7)) # get all elements which are not a filler value
                gru4rec_scores = gru4rec_scores[torch.all(fillval_mask, dim=1)]
                gru4rec_items = gru4rec_items[torch.all(fillval_mask, dim=1)]
                gru4rec_items = gru4rec_items.astype(int)
                ex2vec_scores = ex2vec_scores[torch.all(fillval_mask, dim=1)]
                out_idxs = torch.tensor(out_idxs[out_idxs != -7], dtype=torch.int32)



                #print("Cleaned GRU4Rec Scores: ", gru4rec_scores[45:55, :5])
                #print("Cleaned GRU4Rec Items: ", gru4rec_items[45:55, :5])
                #print("Cleaned Ex2Vec Scores: ", ex2vec_scores[45:55, :5])
                #print("Cleaned out_idxs: ", out_idxs[0:55])

                combined_scores = gru4rec_utils.combine_scores(torch.tensor(gru4rec_scores).cuda(), torch.tensor(ex2vec_scores).cuda(), alpha, combination_mode, 0.5)
                combined_scores = combined_scores.squeeze(0) # remove alpha dimension as we do experiments with single element

                #print("Combined scores: ", combined_scores[45:55,:5])

                # adapted code from original GRU4Rec repo
                oscores = torch.tensor(gru4rec_scores.T) # get original GRU4Rec scores
                oscores_combined = torch.tensor(combined_scores.T.cpu()) # get original combined scores

                #print("oscores: ", oscores[45:55,:5])
                #print("oscores combined: ", oscores_combined[45:55,:5])

                tscores = [] # store true non-comb scores
                tscores_combined = [] # store true comb scores
                for i, out_idx in enumerate(out_idxs): # loop through target items of each input item
                    out_idx_int = out_idx.item()
                    #print("Curr outitem: ", out_idx)
                    #print("Curr topk item row: ", gru4rec_items[i])
                    if out_idx_int in gru4rec_items[i]: # check if out_idx is in topk predicted out_idx
                        out_idx_index = (gru4rec_items[i] == out_idx_int).nonzero()[0].item() # get first index where out_idx is located in gru4rec_items
                        tscores.append(gru4rec_scores[i][out_idx_index]) # retrieve and append corresponding score of the out_idx item in gru4rec scores
                        tscores_combined.append(combined_scores[i][out_idx_index]) # retrieve same score in combined_scores
                    else:
                        # append invalid scores
                        tscores.append(-7)
                        tscores_combined.append(-7)

                # convert lists to tensors
                tscores, tscores_combined = torch.tensor(tscores), torch.tensor(tscores_combined)

                if eval_mode == 'standard':
                    ranks = (oscores > tscores).sum(dim=0) + 1 # calculates how many of the 879 item scores (for one particular in_idx) have scored higher than the corresponding true item (corresponding out_idx) + 1 for actual placement (1st place instead of 0th place)
                    ranks_combined = (oscores_combined > tscores_combined).sum(dim=0) + 1
                elif eval_mode == 'conservative':
                    ranks = (oscores >= tscores).sum(dim=0)
                    ranks_combined = (oscores_combined >= tscores_combined).sum(dim=0)
                elif eval_mode == 'median':
                    ranks = (oscores >= tscores).sum(dim=0) + 0.5*((oscores == tscores).dim(axis=0) - 1) + 1
                    ranks_combined = (oscores_combined >= tscores_combined).sum(dim=0) + 0.5*((oscores_combined == tscores_combined).dim(axis=0) - 1) + 1
                else: raise NotImplementedError

                #print("ranks: ", ranks)
                #print("ranks_combined: ", ranks_combined)

                # calculate batch mrr and recall
                for c in cutoffs:
                    recall[c] += (ranks <= c).sum().cpu().numpy() #e.g. if cutoff==5, then it counts how often the most relevant item is among the top 5  ranks
                    mrr[c] += ((ranks <= c) / ranks.float()).sum().cpu().numpy()
                    recall_combined[c] += (ranks_combined <= c).sum().cpu().numpy() #e.g. if cutoff==5, then it counts how often the most relevant item is among the top 5  ranks
                    mrr_combined[c] += ((ranks_combined <= c) / ranks.float()).sum().cpu().numpy()
                #print("Recall: ", recall)
                #print("Recall combined: ", recall_combined)

                # keep track of batchsize for calculating metrics over all batches
                n += gru4rec_scores.shape[0] # n += batch_size

                pbar.update(1) # update progress bar

        for ca in cutoffs:
            recall[ca] /= n # avg recall over all processed batches
            mrr[ca] /= n # avg mrr over all processed batches
            recall_combined[ca] /= n 
            mrr_combined[ca] /= n 
        return recall, mrr, recall_combined, mrr_combined


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
            oscores = O.T # (879,50), (k, batch_size)
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
