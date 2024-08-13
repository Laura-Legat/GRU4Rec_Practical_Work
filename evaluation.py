from gru4rec_pytorch import SessionDataIterator
import torch

@torch.no_grad()
def batch_eval(gru, test_data, cutoff=[20], batch_size=512, mode='conservative', item_key='ItemId', session_key='SessionId', time_key='Time'):
    if gru.error_during_train: 
        raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
    recall = dict()
    mrr = dict()
    for c in cutoff:
        recall[c] = 0
        mrr[c] = 0
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    n = 0
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
    data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key, session_key, time_key, device=gru.device, itemidmap=gru.data_iterator.itemidmap)
    for in_idxs, out_idxs in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
        for h in H: h.detach_()
        O = gru.model.forward(in_idxs, H, None, training=False)
        oscores = O.T
        tscores = torch.diag(oscores[out_idxs]) # select scores of the true (t(rue)scores) items in the sequence
        if mode == 'standard': ranks = (oscores > tscores).sum(dim=0) + 1
        elif mode == 'conservative': ranks = (oscores >= tscores).sum(dim=0)
        elif mode == 'median':  ranks = (oscores > tscores).sum(dim=0) + 0.5*((oscores == tscores).dim(axis=0) - 1) + 1
        else: raise NotImplementedError
        for c in cutoff:
            recall[c] += (ranks <= c).sum().cpu().numpy()
            mrr[c] += ((ranks <= c) / ranks.float()).sum().cpu().numpy()
        n += O.shape[0]
    for c in cutoff:
        recall[c] /= n
        mrr[c] /= n
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