import json
import torch

def convert_to_param_str(best_param_path):
  """
  Helper function that converts parameters from a JSON file into a string used for training models.

  Args:
    best_param_path: The path of the JSON file containing parameters for model training.

  Returns:
    String of parameters of the form "param1=value1,param2=value2,..."
  """
  with open(best_param_path, 'r') as f:
    data = json.load(f)

  params = data['best_params'] # extract only the parameter part, not the optuna n_trials
  return ','.join([f'{key}={value}' for key,value in params.items()])

def get_itemId(gru, idx_list):
    """
      Maps GRU4Rec-specific indices back to item IDs.

      Args:
          idx_list: item indices of GRU4Rec itemid map -> List
      Returns:
          Item IDs which corresponds to the index argument -> List
    """
    return [gru.data_iterator.itemidmap.index[gru.data_iterator.itemidmap.iloc[idx]] for idx in idx_list]

def normalize_scores(original_scores):
    """
    Applies min-max normalization to a score tensor.

    Args:
        original_scores: The tensor of scores to noralize

    Returns:
        Normalized scores as tensor
    """
    # min-max normalization formula: score - min(score) / max(score) - min(score)
    score_min = original_scores.min(dim=1, keepdim=True)[0]
    score_max = original_scores.max(dim=1, keepdim=True)[0]
    return (original_scores - score_min) / (score_max - score_min + 1e-8) # add small eps to avoid division by 0

def combine_scores(gru4rec_scores, ex2vec_scores, alpha_list, mode = 'direct', ex2vec_threshold=None):
    """
    Combines GRU4Rec and Ex2vec scores depending on the combination mode given.

    Args:
        gru4rec_scores: Contains the corresponding scores for the items -> Tensor (batch_size, n_items) -> tensor([[scores], [scores], [scores], ...])
        ex2vec_scores: The corresponding scores for each itemin gru4rec_items -> Tensor (batch_size, n_items) -> tensor([[scores], [scores], [scores], ...])
        alpha_list: List of alphas to try. Alpha is a parameter set for weighted/boosted combination of scores, i.e. how much to take each models' predictions into account for the final score
        mode: The combination modality (str in [direct, weighted, boosted, mult])
        ex2vec_threshold: Threshold of the scores to include in combination calculations

    Returns:
        List of lists of lists of combined score depending on combination mode and alpha -> [[[gruscores(0)&&ex2vecscores(0) for alpha(0)], [gruscores(0)&&ex2vecscores(0) for alpha(1)], ...],  [[gruscores(1)&&ex2vecscores(1) for alpha(0)], [gruscores(1)&&ex2vecscores(1) for alpha(1), ...], ...]
    """
    #check for enabled negative sampling
    n_sample = gru4rec_scores.shape[1] - ex2vec_scores.shape[1]

    if n_sample > 0:
        gru4rec_scores_positive = gru4rec_scores[:, :ex2vec_scores.shape[1]] # slice off positive samples -> (batch_size, batch_size)
        gru4rec_scores_negative = gru4rec_scores[:, ex2vec_scores.shape[1]:] # slice off negative samples
        gru4rec_scores = gru4rec_scores_positive # enable combination only between positive scores and ex2vec scores

    num_alphas = len(alpha_list)
    # Create a tensor for alphas to avoid repeated tensor creation
    alphas_tensor = torch.tensor(alpha_list, device=gru4rec_scores.device).view(num_alphas, 1, 1)

    if mode == 'direct':
        # zero out scores where ex2vec_scores <= 0.5
        if ex2vec_threshold: ex2vec_scores = ex2vec_scores * (ex2vec_scores >= ex2vec_threshold)
        combined_scores = ex2vec_scores.unsqueeze(0).repeat(num_alphas, 1, 1)
    elif mode == 'weighted':
        # zero out scores where ex2vec_scores <= 0.5
        if ex2vec_threshold: ex2vec_scores = ex2vec_scores * (ex2vec_scores >=  ex2vec_threshold)
        combined_scores = (alphas_tensor * gru4rec_scores) + ((1 - alphas_tensor) * ex2vec_scores)
    elif mode == 'boosted':
        # zero out scores where ex2vec_scores <= 0.5
        if ex2vec_threshold: ex2vec_scores = ex2vec_scores * (ex2vec_scores >= ex2vec_threshold)
        combined_scores = gru4rec_scores + (alphas_tensor * ex2vec_scores)
    elif mode == 'mult':
        # zero out scores where ex2vec_scores <= 0.5
        if ex2vec_threshold: ex2vec_scores = ex2vec_scores * (ex2vec_scores >= threshold)
        combined_scores = (gru4rec_scores * ex2vec_scores).repeat(num_alphas, 1, 1)
    else:
        raise NotImplementedError
    
    if n_sample > 0:
        return torch.cat([combined_scores, gru4rec_scores_negative.unsqueeze(0).repeat(num_alphas, 1, 1)], dim=2) #concat combined positive scores back with old negative scores
    else:
        return combined_scores

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
    # min-max normalization of scores
    gru4rec_scores = normalize_scores(gru4rec_scores)
    ex2vec_scores = normalize_scores(ex2vec_scores)
    
    # get the new, combined scores for each alpha
    combined_scores= combine_scores(gru4rec_scores, ex2vec_scores, alpha_list, mode)

    # define structure for reranked item list
    reranked_items_all_alpha = []
    for i in range(combined_scores.shape[0]):  # combined_scores shape: (num_alphas, batch_size, k), thus loop through alphas
        curr_combined_scores = combined_scores[i]  # shape: (batch_size, k)

        # sort the next-item combined-score tensors based on their highest score, and extract the index that item had previously
        sorted_indices = torch.argsort(curr_combined_scores, dim=1, descending=True)

        # rerank the gru4rec items based on the sorted_indices (highest scores)
        reranked_items = torch.gather(gru4rec_items, 1, sorted_indices)

        # add  reranked items for current alpha to the final list
        reranked_items_all_alpha.append(reranked_items)
    return reranked_items_all_alpha
