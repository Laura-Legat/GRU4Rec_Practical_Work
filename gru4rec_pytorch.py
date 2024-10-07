import math
import numpy as np
import pandas as pd
import torch
from torch import autograd, nn
from torch.autograd import Variable
from collections import OrderedDict
import time
import os
import sys
from tqdm import tqdm
from GRU4Rec_Fork import gru4rec_utils
from data_sampler import get_rel_int_dict


from torch.optim import Optimizer
class IndexedAdagradM(Optimizer):

    def __init__(self, params, lr=0.05, momentum=0.0, eps=1e-6):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if eps <= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super(IndexedAdagradM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['acc'] = torch.full_like(p, 0, memory_format=torch.preserve_format)
                if momentum > 0: state['mom'] = torch.full_like(p, 0, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['acc'].share_memory_()
                if group['momentum'] > 0: state['mom'].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                clr = group['lr']
                momentum = group['momentum']
                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_indices = grad._indices()[0]
                    grad_values = grad._values()
                    accs = state['acc'][grad_indices] + grad_values.pow(2)
                    state['acc'].index_copy_(0, grad_indices, accs)
                    accs.add_(group['eps']).sqrt_().mul_(-1/clr)
                    if momentum > 0:
                        moma = state['mom'][grad_indices]
                        moma.mul_(momentum).add_(grad_values / accs)
                        state['mom'].index_copy_(0, grad_indices, moma)
                        p.index_add_(0, grad_indices, moma)
                    else:
                        p.index_add_(0, grad_indices, grad_values / accs)
                else:
                    state['acc'].add_(grad.pow(2))
                    accs = state['acc'].add(group['eps'])
                    accs.sqrt_()
                    if momentum > 0:
                        mom = state['mom']
                        mom.mul_(momentum).addcdiv_(grad, accs, value=-clr)
                        p.add_(mom)
                    else:
                        p.addcdiv_(grad, accs, value=-clr)
        return loss

def init_parameter_matrix(tensor: torch.Tensor, dim0_scale: int = 1, dim1_scale: int = 1):
    sigma = math.sqrt(6.0 / float(tensor.size(0) / dim0_scale + tensor.size(1) / dim1_scale))
    return nn.init._no_grad_uniform_(tensor, -sigma, sigma)

class GRUEmbedding(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GRUEmbedding, self).__init__()
        self.Wx0 = nn.Embedding(dim_in, dim_out * 3, sparse=True)
        self.Wrz0 = nn.Parameter(torch.empty((dim_out, dim_out * 2), dtype=torch.float))
        self.Wh0 = nn.Parameter(torch.empty((dim_out, dim_out * 1), dtype=torch.float))
        self.Bh0 = nn.Parameter(torch.zeros(dim_out * 3, dtype=torch.float))
        self.reset_parameters()
    def reset_parameters(self):
        init_parameter_matrix(self.Wx0.weight, dim1_scale = 3)
        init_parameter_matrix(self.Wrz0, dim1_scale = 2)
        init_parameter_matrix(self.Wh0, dim1_scale = 1)
        nn.init.zeros_(self.Bh0)
    def forward(self, X, H):
        Vx = self.Wx0(X) + self.Bh0 
        Vrz = torch.mm(H, self.Wrz0)
        vx_x, vx_r, vx_z = Vx.chunk(3, 1)
        vh_r, vh_z = Vrz.chunk(2, 1)
        r = torch.sigmoid(vx_r + vh_r)
        z = torch.sigmoid(vx_z + vh_z) # sigmoid()
        h = torch.tanh(torch.mm(r * H, self.Wh0) + vx_x)
        h = (1.0 - z) * H + z * h
        return h

class GRU4RecModel(nn.Module):
    def __init__(self, n_items, layers=[100], dropout_p_embed=0.0, dropout_p_hidden=0.0, embedding=0, constrained_embedding=True):
        super(GRU4RecModel, self).__init__()
        self.n_items = n_items # n of unique items
        self.layers = layers # n of hidden layers
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.embedding = embedding # embedding size for items
        self.constrained_embedding = constrained_embedding # whether or not to use constrained embedding
        self.start = 0 # layer start index
        if constrained_embedding:
            n_input = layers[-1] # set input size to size of last hidden layer
        elif embedding > 0:
            self.E = nn.Embedding(n_items, embedding, sparse=True)
            n_input = embedding
        else:
            self.GE = GRUEmbedding(n_items, layers[0])
            n_input = n_items
            self.start = 1 # start processing from second layer ??

        self.DE = nn.Dropout(dropout_p_embed) # set dropout for embeddings

        self.G = [] # store GRU cells
        self.D = [] # store dropout layers

        for i in range(self.start, len(layers)): # for each layer in the GRU
            self.G.append(nn.GRUCell(layers[i-1] if i > 0 else n_input, layers[i])) # new GRU layer for each layer in layers is created - input size is n_input at layer[0], for all other layers it is the size of the output of the previous layer
            self.D.append(nn.Dropout(dropout_p_hidden)) # add dropout layer with certain drop prob

        # store cells and dropout layers as nn.Modules in a list
        self.G = nn.ModuleList(self.G)
        self.D = nn.ModuleList(self.D)

        self.Wy = nn.Embedding(n_items, layers[-1], sparse=True)
        self.By = nn.Embedding(n_items, 1, sparse=True)
        
        self.reset_parameters() # init all model params

    @torch.no_grad()
    def reset_parameters(self):
        if self.embedding > 0:
            init_parameter_matrix(self.E.weight)
        elif not self.constrained_embedding:
            self.GE.reset_parameters()
        for i in range(len(self.G)):
            init_parameter_matrix(self.G[i].weight_ih, dim1_scale = 3)
            init_parameter_matrix(self.G[i].weight_hh, dim1_scale = 3)
            nn.init.zeros_(self.G[i].bias_ih)
            nn.init.zeros_(self.G[i].bias_hh)
        init_parameter_matrix(self.Wy.weight)
        nn.init.zeros_(self.By.weight)
    def _init_numpy_weights(self, shape):
        sigma = np.sqrt(6.0 / (shape[0] + shape[1]))
        m = np.random.rand(*shape).astype('float32') * 2 * sigma - sigma
        return m
    @torch.no_grad()
    def _reset_weights_to_compatibility_mode(self):
        np.random.seed(42)
        if self.constrained_embedding:
            n_input = self.layers[-1]
        elif self.embedding:
            n_input = self.embedding
            self.E.weight.set_(torch.tensor(self._init_numpy_weights((self.n_items, n_input)), device=self.E.weight.device))
        else:
            n_input = self.n_items
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            self.GE.Wx0.weight.set_(torch.tensor(np.hstack(m), device=self.GE.Wx0.weight.device))
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[0] , self.layers[0])))
            m2.append(self._init_numpy_weights((self.layers[0] , self.layers[0])))
            self.GE.Wrz0.set_(torch.tensor(np.hstack(m2), device=self.GE.Wrz0.device))
            self.GE.Wh0.set_(torch.tensor(self._init_numpy_weights((self.layers[0] , self.layers[0])), device=self.GE.Wh0.device))
            self.GE.Bh0.set_(torch.zeros((self.layers[0]*3,), device=self.GE.Bh0.device))
        for i in range(self.start, len(self.layers)):
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            self.G[i].weight_ih.set_(torch.tensor(np.vstack(m), device=self.G[i].weight_ih.device))
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[i] , self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i] , self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i] , self.layers[i])))
            self.G[i].weight_hh.set_(torch.tensor(np.vstack(m2), device=self.G[i].weight_hh.device))
            self.G[i].bias_hh.set_(torch.zeros((self.layers[i]*3,), device=self.G[i].bias_hh.device))
            self.G[i].bias_ih.set_(torch.zeros((self.layers[i]*3,), device=self.G[i].bias_ih.device))
        self.Wy.weight.set_(torch.tensor(self._init_numpy_weights((self.n_items, self.layers[-1])), device=self.Wy.weight.device))
        self.By.weight.set_(torch.zeros((self.n_items, 1), device=self.By.weight.device))
    def embed_constrained(self, X, Y=None):
        if Y is not None: #training
            XY = torch.cat([X, Y]) # concatinate in_idx and out_idx -> shape (batch_size + (batch_size+n_sample))
            EXY = self.Wy(XY) # retrieve item embds for concat version of in_idx and out_idx (batch_size + (batch_size+n_sample), emb_dim)
            split = X.shape[0] # define a split point for splitting them according to X and Y shapes again
            E = EXY[:split] # embedding for X/in_idx -> (batch_size, emb_dim)
            O = EXY[split:] # embedding for Y/out_idx -> (batch_size + n_sample, emb_dim)
            B = self.By(Y) # bias for Y/out_idx -> (batch_size + n_sample, 1)
        else: # eval
            E = self.Wy(X) # compute embeddings for X
            O = self.Wy.weight # weights for X
            B = self.By.weight # bias for X
        return E, O, B
    def embed_separate(self, X, Y=None):
        E = self.E(X) # embeds for X
        if Y is not None: # training
            O = self.Wy(Y) # embeds for Y -> (batch_size+n_sample, emb_dim)
            B = self.By(Y) # biases for Y -> (batch_size + n_sample, 1)
        else: #eval
            O = self.Wy.weight # embeds for all items
            B = self.By.weight # bias for all items
        return E, O, B
    def embed_gru(self, X, H, Y=None):
        E = self.GE(X, H) # embeds for X
        if Y is not None:
            O = self.Wy(Y) # embeds for Y -> (batch_size+n_sample, emb_dim)
            B = self.By(Y) # biases for Y -> (batch_size + n_sample, 1)
        else:
            O = self.Wy.weight # embedding weights for all items = X
            B = self.By.weight # biases for all items
        return E, O, B
    def embed(self, X, H, Y=None):
        if self.constrained_embedding:
            E, O, B = self.embed_constrained(X, Y)
        elif self.embedding > 0:
            E, O, B = self.embed_separate(X, Y)
        else:
            E, O, B = self.embed_gru(X, H[0], Y)
        return E, O, B
    def hidden_step(self, X, H, training=False):
        """
        Args:
            X: input tensor, embeddings of items
            H: Hidden state tensor, one hidden state for each GRU layer
            training: If model is in training mode or not
        """
        for i in range(self.start, len(self.layers)): # iterate over GRU layer (there is only 1)
            X = self.G[i](X, Variable(H[i])) # pass input x_t and hidden state through GRU cell

            if training:
                X = self.D[i](X) # pass input through dropout, set some values to 0

            H[i] = X # store new hidden state (X which GRU returns) back into H[i], updating the hidden state for layer i 
        return X
    def score_items(self, X, O, B):
        O = torch.mm(X, O.T) + B.T # X * H + Y (batch_size, emb_dim) * (emb_dim, batch_size + n_sample) = (batch_size, batch_size+n_sample) + (1, batch_size+n_sample) = (batch_size, batch_size+n_sample)
        return O
    
    def forward(self, X, H, Y, training=False):
        """
        Embedding -> Dropout Embd -> Pass through GRU layers -> scoring

        Args:
            X: Input tensor, in_idxs
            H: Hidden state tensor
            Y: Target tensor, target item indices, out_idxs
            training: If model is in training mode or not
        """
        E, O, B = self.embed(X, H, Y)

        if training: 
            E = self.DE(E) # dropout - set some of embedding values of in_idx to 0 during training

        if not (self.constrained_embedding or self.embedding):
            H[0] = E

        Xh = self.hidden_step(E, H, training=training) # pass embeddings E through stracked GRU layers
        R = self.score_items(Xh, O, B) # get relevance scores
        return R

class SampleCache:
    def __init__(self, n_sample, sample_cache_max_size, distr, device=torch.device('cuda:0')):
        self.device = device
        self.n_sample = n_sample
        self.generate_length = sample_cache_max_size // n_sample if n_sample > 0 else 0
        self.distr = distr
        self._refresh()
        print('Created sample store with {} batches of samples (type=GPU)'.format(self.generate_length))
    def _bin_search(self, arr, x):
        l = x.shape[0]
        a = torch.zeros(l, dtype=torch.int64, device=self.device)
        b = torch.zeros(l, dtype=torch.int64, device=self.device) + arr.shape[0]
        while torch.any(a != b):
            ab = torch.div((a + b), 2,  rounding_mode='trunc')
            val = arr[ab]
            amask = (val <= x)
            a[amask] = ab[amask] + 1
            b[~amask] = ab[~amask]
        return a
    def _refresh(self):
        if self.n_sample <= 0: return
        x = torch.rand(self.generate_length * self.n_sample, dtype=torch.float32, device=self.device)
        self.neg_samples = self._bin_search(self.distr, x).reshape((self.generate_length, self.n_sample))
        self.sample_pointer = 0
    def get_sample(self):
        if self.sample_pointer >= self.generate_length:
            self._refresh()
        sample = self.neg_samples[self.sample_pointer]
        self.sample_pointer += 1
        return sample

class SessionDataIterator:
    def __init__(self, data, batch_size, n_sample=0, sample_alpha=0.75, sample_cache_max_size=10000000, item_key='ItemId', user_key = 'userId', session_key='SessionId', rel_int_key="relational_interval", time_key='Time', session_order='sequential', device=torch.device('cuda:0'), itemidmap=None):
        """
        Args:
            data: Sessionized input dataset -> Pandas DataFrame
            batch_size: Desired batch size to be used during training -> int
            n_sample: Number of negative samples to use -> int
            sample_alpha: Distribution of negative samples during negative sampling -> float
            session_order: 'time' or 'sequential'. Setting it to 'time' means that sessions are sorted and processed according to increasing timestamps -> str
            itemidmap: An optional mapping of item IDs to indices

        """
        self.device = device
        self.batch_size = batch_size

        # map userids according to ex2vec
        userids = data['userId'].unique()
        user_mapping = pd.Series(data=np.arange(len(userids), dtype='int32'), index=userids, name='userIdx')
        data['userId'] = data['userId'].map(user_mapping)

        self.data = data

        if itemidmap is None:
            print('Creating new item ID map')
            itemids = data[item_key].unique() # array of unique item IDs
            self.n_items = len(itemids) # counts total amount of unique items in data
            # create mapping between item IDs and indices
            self.itemidmap = pd.Series(data=np.arange(self.n_items, dtype='int32'), index=itemids, name='ItemIdx') # map item IDs to indices from 0 to self.n_items - 1; index=itemids associates each index with corresponding item ID
        else:
            print('Using existing item ID map')
            self.itemidmap = itemidmap
            self.n_items = len(itemidmap)
            in_mask = data[item_key].isin(itemidmap.index.values)
            n_not_in = (~in_mask).sum()
            if n_not_in > 0:
                #print('{} rows of the data contain unknown items and will be filtered'.format(n_not_in))
                data = data.drop(data.index[~in_mask])
        self.sort_if_needed(data, [session_key]) # sort by session key
        self.offset_sessions = self.compute_offset(data, session_key) # calculate starting indices of each session

        if session_order == 'time':
            self.session_idx_arr = np.argsort(data.groupby(session_key)[time_key].min().values) # create array of smallest timestamps per session, argsort then gives an order which session should be processed first, which second etc. 
        else:
            self.session_idx_arr = np.arange(len(self.offset_sessions) - 1)

        self.data_items = self.itemidmap[data[item_key].values].values # item indices are stored

        if n_sample > 0: # negative sampling
            pop = data.groupby(item_key).size() # how many user-item interactions of each item are in the data
            pop = pop[self.itemidmap.index.values].values**sample_alpha # assigns popularity to indices
            pop = pop.cumsum() / pop.sum() # normalize popularity values -> calc probability for each item that this item is chosen as a negative sample
            pop[-1] = 1
            distr = torch.tensor(pop, device=self.device, dtype=torch.float32)
            self.sample_cache = SampleCache(n_sample, sample_cache_max_size, distr, device=self.device) # cache to store pre-sampled negative items -> efficiently retrieve negative samples

    def sort_if_needed(self, data, columns, any_order_first_dim=False):
        is_sorted = True
        neq_masks = []
        for i, col in enumerate(columns):
            dcol = data[col]
            neq_masks.append(dcol.values[1:]!=dcol.values[:-1])
            if i == 0:
                if any_order_first_dim:
                    is_sorted = is_sorted and (dcol.nunique() == neq_masks[0].sum() + 1)
                else:
                    is_sorted = is_sorted and np.all(dcol.values[1:] >= dcol.values[:-1])
            else:
                is_sorted = is_sorted and np.all(neq_masks[i - 1] | (dcol.values[1:] >= dcol.values[:-1]))
            if not is_sorted:
                break
        if is_sorted:
            print('The dataframe is already sorted by {}'.format(', '.join(columns)))
        else:
            print('The dataframe is not sorted by {}, sorting now'.format(col))
            t0 = time.time()
            data.sort_values(columns, inplace=True)
            t1 = time.time()
            print('Data sorting took {:.2f}s'.format(t1 - t0))

    def compute_offset(self, data, column):
        offset = np.zeros(data[column].nunique() + 1, dtype=np.int32)
        offset[1:] = data.groupby(column).size().cumsum()
        return offset

    def __call__(self, enable_neg_samples, reset_hook=None):
        batch_size = self.batch_size
        iters = np.arange(batch_size) # len batch_size
        maxiter = iters.max()
        start = self.offset_sessions[self.session_idx_arr[iters]]
        end = self.offset_sessions[self.session_idx_arr[iters]+1]
        finished = False
        valid_mask = np.ones(batch_size, dtype='bool')
        n_valid = self.batch_size
        while not finished:
            minlen = (end-start).min()
            out_idx = torch.tensor(self.data_items[start], requires_grad=False, device=self.device)
            for i in range(minlen-1):
                in_idx = out_idx
                out_idx = torch.tensor(self.data_items[start+i+1], requires_grad=False, device=self.device)

                # extract current user and sessionid to make it easier for ex2vec inference
                curr_users = []
                curr_sessions = []
                curr_rel_ints = []
                for j in range(n_valid): #for j in batch_size
                    curr_users.append(self.data.iloc[start[j]]['userId'])
                    curr_sessions.append(self.data.iloc[start[j]]['SessionId'])
                    curr_rel_ints.append(self.data.iloc[start[j]]['relational_interval'])

                if enable_neg_samples:
                    sample = self.sample_cache.get_sample() # Tensor -> (n_sample), sample n_sample item IDs
                    y_positive = out_idx # store positive samples (batch_size)
                    y = torch.cat([out_idx, sample]) # out_idx -> (batch_size + n_sample)
                else:
                    y_positive = out_idx
                    y = out_idx
                yield in_idx, y, curr_users, curr_sessions, curr_rel_ints, y_positive
            start = start+minlen-1
            finished_mask = (end-start<=1)
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1,n_finished+1)
            maxiter += n_finished
            valid_mask = (iters < len(self.offset_sessions)-1)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                finished = True
                break
            mask = finished_mask & valid_mask
            sessions = self.session_idx_arr[iters[mask]]
            start[mask] = self.offset_sessions[sessions]
            end[mask] = self.offset_sessions[sessions+1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]
            if reset_hook is not None:
                finished = reset_hook(n_valid, finished_mask, valid_mask)

class GRU4Rec:
    def __init__(self, layers=[100], loss='cross-entropy', batch_size=64, dropout_p_embed=0.0,
                 dropout_p_hidden=0.0, learning_rate=0.05, momentum=0.0, sample_alpha=0.5, n_sample=2048, embedding=0,
                 constrained_embedding=True, n_epochs=10, bpreg=1.0, elu_param=0.5, logq=0.0, device=torch.device('cuda:0')):
        self.device = device
        self.layers = layers
        self.loss = loss
        self.set_loss_function(loss)
        self.elu_param = elu_param
        self.bpreg = bpreg
        self.logq = logq
        self.batch_size = batch_size
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.sample_alpha = sample_alpha
        self.n_sample = n_sample
        if embedding == 'layersize':
            self.embedding = self.layers[0]
        else:
            self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.n_epochs = n_epochs
    def set_loss_function(self, loss):
        if loss == 'cross-entropy': self.loss_function = self.xe_loss_with_softmax
        elif loss == 'bpr-max': self.loss_function = self.bpr_max_loss_with_elu
        else: raise NotImplementedError
    def set_params(self, **kvargs):
        maxk_len = np.max([len(str(x)) for x in kvargs.keys()])
        maxv_len = np.max([len(str(x)) for x in kvargs.values()])
        for k,v in kvargs.items():
            if not hasattr(self, k):
                print('Unkown attribute: {}'.format(k))
                raise NotImplementedError
            else:
                if type(v) == str and type(getattr(self, k)) == list: v = [int(l) for l in v.split('/')]
                if type(v) == str and type(getattr(self, k)) == bool:
                    if v == 'True' or v == '1': v = True
                    elif v == 'False' or v == '0': v = False
                    else:
                        print('Invalid value for boolean parameter: {}'.format(v))
                        raise NotImplementedError
                if k == 'embedding' and v == 'layersize':
                    self.embedding = 'layersize'
                setattr(self, k, type(getattr(self, k))(v))
                if k == 'loss': self.set_loss_function(self.loss)
                print('SET   {}{}TO   {}{}(type: {})'.format(k, ' '*(maxk_len-len(k)+3), getattr(self, k), ' '*(maxv_len-len(str(getattr(self, k)))+3), type(getattr(self, k))))
        if self.embedding == 'layersize':
            self.embedding = self.layers[0]
            print('SET   {}{}TO   {}{}(type: {})'.format('embedding', ' '*(maxk_len-len('embedding')+3), getattr(self, 'embedding'), ' '*(maxv_len-len(str(getattr(self, 'embedding')))+3), type(getattr(self, 'embedding'))))
    def xe_loss_with_softmax(self, O, Y, M): # cross-entropy
        """
        R, out_idx, n_valid
        """
        if self.logq > 0:
            O = O - self.logq * torch.log(torch.cat([self.P0[Y[:M]], self.P0[Y[M:]]**self.sample_alpha]))
        X = torch.exp(O - O.max(dim=1, keepdim=True)[0])
        X = X / X.sum(dim=1, keepdim=True)
        return -torch.sum(torch.log(torch.diag(X)+1e-24))
    def softmax_neg(self, X):
        hm = 1.0 - torch.eye(*X.shape, out=torch.empty_like(X))
        X = X * hm
        e_x = torch.exp(X - X.max(dim=1, keepdim=True)[0]) * hm
        return e_x / e_x.sum(dim=1, keepdim=True)

    def bpr_max_loss_with_elu(self, O, Y, M): #bpr-max
        if self.elu_param > 0:
            O = nn.functional.elu(O, self.elu_param)
        softmax_scores = self.softmax_neg(O)
        target_scores = torch.diag(O)
        target_scores = target_scores.reshape(target_scores.shape[0],-1)

        # BPR formula(?) pp. 5 in gru4rec paper + term for stabilization (1e-24) + regularization term +self.bpreg*torch.sum((O**2)*softmax_scores, dim=1)
        return torch.sum((-torch.log(torch.sum(torch.sigmoid(target_scores-O)*softmax_scores, dim=1)+1e-24)+self.bpreg*torch.sum((O**2)*softmax_scores, dim=1)))
    
    def fit(self, data, sample_cache_max_size=10000000, compatibility_mode=True, item_key='ItemId', session_key='SessionId', time_key='Time', combination=None, ex2vec=None, alpha=[0.2], save_path=None): # Training loop of the model
        self.error_during_train = False # flag for tracking NaN losses during training

        self.data_iterator = SessionDataIterator(data, self.batch_size, n_sample=self.n_sample, sample_alpha=self.sample_alpha, sample_cache_max_size=sample_cache_max_size, item_key=item_key, session_key=session_key, time_key=time_key, session_order='sequential', device=self.device) # iterator to loop over sessionized data
        # logq flag (either 0 or 1) only has effect with cross-entropy loss
        if self.logq and self.loss == 'cross-entropy':
            pop = data.groupby(item_key).size() # counts popularity of item by simple counting how many interactions it has
            self.P0 = torch.tensor(pop[self.data_iterator.itemidmap.index.values], dtype=torch.float32, device=self.device) # make a tensor out of it ready for passing to loss calculation

        model = GRU4RecModel(self.data_iterator.n_items, self.layers, self.dropout_p_embed, self.dropout_p_hidden, self.embedding, self.constrained_embedding).to(self.device) # init GRU4Rec model and move to GPU

        if compatibility_mode: 
            model._reset_weights_to_compatibility_mode()

        self.model = model

        opt = IndexedAdagradM(self.model.parameters(), self.learning_rate, self.momentum) # init optimizer

        if combination != None:
            # put ex2vec into eval mode
            ex2vec.model.eval()

            # get copy of global rel ints
            rel_int_dict = get_rel_int_dict().copy()

            # update global rel_int with current listenings
            for item, user, rel_int in zip(data['itemId'], data['userId'], data['relational_interval']):
                # update the global rel_ints dict for the current user-item interaction
                rel_int_dict[(user, item)] = rel_int

        total_batches = len(data) // self.batch_size  # Calculate total number of batches
        if len(data) % self.batch_size != 0:
            total_batches += 1

        for epoch in range(self.n_epochs): # training loop
            t0 = time.time()
            H = [] # init hidden state for each layer

            for i in range(len(self.layers)):
                H.append(torch.zeros((self.batch_size, self.layers[i]), dtype=torch.float32, requires_grad=False, device=self.device))

            c = [] # track losses
            cc = [] # track number of valid samples
            n_valid = self.batch_size
            reset_hook = lambda n_valid, finished_mask, valid_mask: self._adjust_hidden(n_valid, finished_mask, valid_mask, H) # adjust hidden state when session ends

            with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{self.n_epochs}', unit='batch', ncols=100) as pbar:
              for in_idx, out_idx, userids, sess_id, rel_ints, next_positive in self.data_iterator(enable_neg_samples=(self.n_sample>0), reset_hook=reset_hook):
                  for h in H: h.detach_() # detach hidden states to avoid gradient accumulation from prev batch

                  self.model.zero_grad() 
                  R = self.model.forward(in_idx, H, out_idx, training=True) # -> (batch_size, batch_size + negative_sample)

                  if combination != None:
                      rel_ints_next = [rel_int_dict.get((user, item), []) for user, item in zip(userids, next_positive.tolist())]

                      # scoring top-k next-item predictions with ex2vec
                      ex2vec_scores, _ = ex2vec.model(torch.tensor(np.array(userids), device=self.device), next_positive.to(self.device), torch.tensor(np.array([np.pad(rel_int, (0, 50 - len(rel_int)), constant_values=-1) for rel_int in rel_ints_next]), device=self.device))
                      ex2vec_scores_aligned = ex2vec_scores.repeat(R.shape[0], 1)

                      # calculate new next-item scores depending on combination with ex2vec
                      R = gru4rec_utils.combine_scores(R, ex2vec_scores_aligned, alpha, combination).squeeze(0)

                  # loss per batch / batch_size = avg loss per sample
                  L = self.loss_function(R, out_idx, n_valid) / self.batch_size

                  L.backward() # computes grads
                  opt.step() # update model params using the opimizer

                  L = L.cpu().detach().numpy()
                  c.append(L)
                  cc.append(n_valid)

                  # NaN loss values -> logging the error
                  if np.isnan(L):
                      print(str(epoch) + ': NaN error!')
                      self.error_during_train = True
                      return

                  pbar.update(1)  # Increment the progress bar by 1 for each batch
                
            c = np.array(c)
            cc = np.array(cc)
            avgc = np.sum(c * cc) / np.sum(cc) # compute avg loss for epoch

            # handling/logging NaN losses
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            
            t1 = time.time() # log training time
            dt = t1 - t0

            print('Epoch{} --> loss: {:.6f} \t({:.2f}s) \t[{:.2f} mb/s | {:.0f} e/s]'.format(epoch+1, avgc, dt, len(c)/dt, np.sum(cc)/dt))

            # save checkpoint every second epoch
            if epoch % 2 == 0:
                if save_path:
                    chckpt_path = save_path.format(epoch, avgc)
                    self.savemodel(path=chckpt_path)

    def _adjust_hidden(self, n_valid, finished_mask, valid_mask, H):
        if (self.n_sample == 0) and (n_valid < 2):
            return True
        with torch.no_grad():
            for i in range(len(self.layers)):
                H[i][finished_mask] = 0
        if n_valid < len(valid_mask):
            for i in range(len(H)):
                H[i] = H[i][valid_mask]
        return False
    def to(self, device):
        if type(device) == str:
            device = torch.device(device)
        if device == self.device:
            return
        if hasattr(self, 'model'):
            self.model = self.model.to(device)
            self.model.eval()
        self.device = device
        if hasattr(self, 'data_iterator'):
            self.data_iterator.device = device
            if hasattr(self.data_iterator, 'sample_cache'):
                self.data_iterator.sample_cache.device = device
        pass
    def savemodel(self, path, param_str = None, metric_str = None, log_path = None, is_checkpoint=True):
        torch.save(self, path)

        if is_checkpoint == False:
            model_name = os.path.basename(path).split('.')[0] # split out model name without .pt ending

            new_row = {
                'model_name': [model_name],
                'item_embds': [''],
                'params': [param_str],
                'results': [metric_str]
            }

            new_row_df = pd.DataFrame(new_row)        
            if os.path.exists(log_path): # append contents without header
                new_row_df.to_csv(log_path, mode='a', header=False, index=False)
            else: # create header and then append contents
                new_row_df.to_csv(log_path, mode='w', header=True, index=False)

            print(f'Final model saved to {path}')
        else:
            print(f'Checkpoint saved to {path}')
    @classmethod
    def loadmodel(cls, path, device='cuda:0'):
        gru = torch.load(path, map_location=device)
        gru.device = torch.device(device)
        if hasattr(gru, 'data_iterator'):
            gru.data_iterator.device = torch.device(device)
            if hasattr(gru.data_iterator, 'sample_cache'):
                gru.data_iterator.sample_cache.device = torch.device(device)
        gru.model.eval()
        return gru