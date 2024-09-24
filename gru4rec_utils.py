import json
def convert_to_param_str(best_param_path):
  with open(best_param_path, 'r') as f:
    data = json.load(f)

  params = data['best_params'] # extract only the parameter part
  param_str = ','.join([f'{key}={value}' for key,value in params.items()])

  return param_str

def get_itemId(gru, idx_list):
    """
      Maps GRU4Rec-specific indices back to item IDs.

      Args:
          idx_list: item indices of GRU4Rec itemid map -> List
      Returns:
          Item IDs which corresponds to the index argument -> List
    """
    return [gru.data_iterator.itemidmap.index[gru.data_iterator.itemidmap.iloc[idx]] for idx in idx_list]

def combine_scores(gru4rec_scores, ex2vec_scores, alpha_list, mode = 'direct'):
    """
    Combines GRU4Rec and Ex2vec scores depending on the combination mode given.

    Args:
        gru4rec_scores: Contains the corresponding scores for the items, list of lists
        ex2vec_scores: The corresponding scores for each itemin gru4rec_items
        alpha_list: List of alphas to try. Alpha is a parameter set for weighted/boosted combination of scores, i.e. how much to take each models' predictions into account for the final score
        mode: The combination modality (str in [direct, weighted, boosted, mult])

    Returns:
        List of lists of lists of combined score depending on combination mode and alpha -> [[[gruscores(0)&&ex2vecscores(0) for alpha(0)], [gruscores(0)&&ex2vecscores(0) for alpha(1)], ...],  [[gruscores(1)&&ex2vecscores(1) for alpha(0)], [gruscores(1)&&ex2vecscores(1) for alpha(1), ...], ...]
    """
    if mode == 'direct':
        return [[ex2vec_score_list for ex2vec_score_list in ex2vec_scores] for alpha in alpha_list]
    elif mode == 'weighted':
        return [[[(alpha * gru4rec_score) + ((1-alpha) * ex2vec_score) for gru4rec_score, ex2vec_score in zip(gru4rec_score_list, ex2vec_score_list)] for gru4rec_score_list, ex2vec_score_list in zip(gru4rec_scores, ex2vec_scores)] for alpha in alpha_list]
    elif mode == 'boosted':
        return [[[gru4rec_score + (alpha * ex2vec_score) for gru4rec_score, ex2vec_score in zip(gru4rec_score_list, ex2vec_score_list)] for gru4rec_score_list, ex2vec_score_list in zip(gru4rec_scores, ex2vec_scores)] for alpha in alpha_list]
    elif mode == 'mult':
        return [[[gru4rec_score * ex2vec_score for gru4rec_score, ex2vec_score in zip(gru4rec_score_list, ex2vec_score_list)] for gru4rec_score_list, ex2vec_score_list in zip(gru4rec_scores, ex2vec_scores)] for alpha in alpha_list]
    else: raise NotImplementedError

def rerank(gru4rec_items, gru4rec_scores, ex2vec_scores, alpha_list, mode = 'direct'):
    """
    Reranks recommended items based on Ex2Vecs scores for those items;

    Args:
        gru4rec_items: List of lists of recommended next items, where each child list contains the top-k recommendations for the next items for one timestep
        gru4rec_scores: Contains the corresponding scores for the items, list of lists
        ex2vec_scores: The corresponding scores for each itemin gru4rec_items
        alpha: Parameter set for weighted/boosted combination of scores, i.e. how much to take each models' predictions into account for the final score
        mode: The combination modality (str in [direct, weighted, boosted, mult])

    Returns:
        reranked_items: List of items in re-ranked order, based on their ex2vec score -> List
    """

    # get the new, combined scores for each alpha
    combined_scores= combine_scores(gru4rec_scores, ex2vec_scores, alpha_list, mode)

    # define structure for reranked item list
    reranked_items_all_alpha = []
    for i, alpha in enumerate(alpha_list):
        reranked_items = []
        curr_combined_scores = combined_scores[i] # get scores for current alpha

        for gru4rec_item_list, gru4rec_score_list, ex2vec_score_list, combined_score_list in zip(gru4rec_items, gru4rec_scores, ex2vec_scores, curr_combined_scores): # loop through inner lists
            # pair each item with its score in inner lists
            item_score_pairs = list(zip(gru4rec_item_list, combined_score_list)) # [(itemid, score), (itemid, score),...]

            # sort pairs descendingly based on score value
            sorted_item_score_pairs = sorted(item_score_pairs, key=lambda x: x[1], reverse=True)

            #extract the items from the pairs
            sorted_items = [item for item, score in sorted_item_score_pairs]
            reranked_items.append(sorted_items)
        reranked_items_all_alpha.append(reranked_items)
    return reranked_items_all_alpha