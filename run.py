import argparse # lib for parsing command-line args
import os
import os.path
import shutil # functionality for working with files
from ast import literal_eval
import numpy as np
import pandas as pd
import datetime as dt
import sys
import time
from collections import OrderedDict
import importlib
import importlib.util
import joblib
from gru4rec_utils import convert_to_param_str
import evaluation
from ex2vec import Ex2VecEngine


# format help message nicely, which is displayed when running python run.py -h
class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Train or load a GRU4Rec model & measure recall and MRR on the specified test set(s).')

# define args the user can set via the command line
parser.add_argument('path', metavar='PATH', type=str, help='Path to the training data (TAB separated file (.tsv or .txt) or pickled pandas.DataFrame object (.pickle)) (if the --load_model parameter is NOT provided) or to the serialized model (if the --load_model parameter is provided).')
parser.add_argument('-ps', '--parameter_string', metavar='PARAM_STRING', type=str, help='Training parameters provided as a single parameter string. The format of the string is `param_name1=param_value1,param_name2=param_value2...`, e.g.: `loss=bpr-max,layers=100,constrained_embedding=True`. Boolean training parameters should be either True or False; parameters that can take a list should use / as the separator (e.g. layers=200/200). Mutually exclusive with the -pf (--parameter_file) and the -l (--load_model) arguments and one of the three must be provided.')
parser.add_argument('-pf', '--parameter_file', metavar='PARAM_PATH', type=str, help='Alternatively, training parameters can be set using a config file specified in this argument. The config file must contain a single OrderedDict named `gru4rec_params`. The parameters must have the appropriate type (e.g. layers = [100]). Mutually exclusive with the -ps (--parameter_string) and the -l (--load_model) arguments and one of the three must be provided.')
parser.add_argument('-l', '--load_model', action='store_true', help='Load an already trained model instead of training a model. Mutually exclusive with the -ps (--parameter_string) and the -pf (--parameter_file) arguments and one of the three must be provided.')
parser.add_argument('-s', '--save_model', metavar='MODEL_PATH', type=str, help='Save the trained model to the MODEL_PATH. (Default: don\'t save model)')
parser.add_argument('-t', '--test', metavar='TEST_PATH', type=str, nargs='+', help='Path to the test data set(s) located at TEST_PATH. Multiple test sets can be provided (separate with spaces). (Default: don\'t evaluate the model)')
parser.add_argument('-m', '--measure', metavar='AT', type=int, nargs='+', default=[20], help='Measure recall & MRR at the defined recommendation list length(s). Multiple values can be provided. (Default: 20)')
parser.add_argument('-e', '--eval_type', metavar='EVAL_TYPE', choices=['standard', 'conservative', 'median'], default='standard', help='Sets how to handle if multiple items in the ranked list have the same prediction score (which is usually due to saturation or an error). See the documentation of batch_eval() in evaluation.py for further details. (Default: standard)')
parser.add_argument('-ss', '--sample_store_size', metavar='SS', type=int, default=10000000, help='GRU4Rec uses a buffer for negative samples during training to maximize GPU utilization. This parameter sets the buffer length. Lower values require more frequent recomputation, higher values use more (GPU) memory. Unless you know what you are doing, you shouldn\'t mess with this parameter. (Default: 10000000)')
parser.add_argument('-g', '--gru4rec_model', metavar='GRFILE', type=str, default='gru4rec_pytorch', help='Name of the file containing the GRU4Rec class. Can be used to select different varaiants. (Default: gru4rec_pytorch)')
parser.add_argument('-d', '--device', metavar='D', type=str, default='cuda:0', help='Device used for computations (default: cuda:0).')
parser.add_argument('-ik', '--item_key', metavar='IK', type=str, default='ItemId', help='Column name corresponding to the item IDs (detault: ItemId).')
parser.add_argument('-sk', '--session_key', metavar='SK', type=str, default='SessionId', help='Column name corresponding to the session IDs (default: SessionId).')
parser.add_argument('-tk', '--time_key', metavar='TK', type=str, default='Time', help='Column name corresponding to the timestamp (default: Time).')
parser.add_argument('-c', '--combination', type=str, default=None, help='How the score of ex2vec and gru4rec should be combined (str in [direct, weighted, boosted, mult]) during inference, for no combination it is None (default: None).')
parser.add_argument('-a', '--alpha', type=float, nargs='+', default=[0.2], help='Alpha value used for combination (only pass one value for training).')
parser.add_argument('-uk', '--user_key', metavar='UK', type=str, default='userId', help='Column name corresponding to the user ID (default: userId).')
parser.add_argument('-rk', '--rel_int_key', metavar='RK', type=str, default='relational_interval', help='Column name corresponding to the relational interval (default: relational_interval).')
parser.add_argument('-pm', '--primary_metric', metavar='METRIC', choices=['recall', 'mrr'], default='recall', help='Set primary metric, recall or mrr (e.g. for paropt). (Default: recall)')
parser.add_argument('-lpm', '--log_primary_metric', action='store_true', help='If provided, evaluation will log the value of the primary metric at the end of the run. Only works with one test file and list length.')
parser.add_argument('--optim', action='store_true', help='Sets the flag that this is hyperparameter tuning instead of normal training.')

# parameters for setting up ex2vec in case of combination
parser.add_argument('-pth', '--base_path', type=str, default='./', help='Base path for Ex2Vec setup (default: ./)')
parser.add_argument('-ex', '--ex2vec_path', type=str, default=None, help='Path to a pre-trained ex2vec model, used as regularization during training (default: None).')
args = parser.parse_args() # parse user-set args and store them in args variable

orig_cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__))) # change dir to where run.py is located, __file__ always refers to the path of the script file being executed
# import GRU4Rec class from module/filename specified in arg/flag "gru4rec_model"
GRU4Rec = importlib.import_module(args.gru4rec_model).GRU4Rec
os.chdir(orig_cwd)

def load_data(fname, args):
    if fname.endswith('.pickle'):
        print('Loading data from pickle file: {}'.format(fname))
        data = joblib.load(fname)
        if args.session_key not in data.columns:
            print('ERROR. The column specified for session IDs "{}" is not in the data file ({})'.format(args.session_key, fname))
            print('The default column name is "SessionId", but you can specify otherwise by setting the `session_key` parameter of the model.')
            sys.exit(1)
        if args.item_key not in data.columns:
            print('ERROR. The column specified for item IDs "{}" is not in the data file ({})'.format(args.item_key, fname))
            print('The default column name is "ItemId", but you can specify otherwise by setting the `item_key` parameter of the model.')
            sys.exit(1)
        if args.time_key not in data.columns:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(args.time_key, fname))
            print('The default column name is "Time", but you can specify otherwise by setting the `time_key` parameter of the model.')
            sys.exit(1)
        if args.user_key not in data.columns:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(args.user_key, fname))
            print('The default column name is "userId", but you can specify otherwise by setting the `user_key` parameter of the model.')
            sys.exit(1)
    else:
        with open(fname, 'rt') as f: # opens file in Read-mode and Text-mode
            header = f.readline().strip().split(',') # get column names
        if args.session_key not in header:
            print('ERROR. The column specified for session IDs "{}" is not in the data file ({})'.format(args.session_key, fname))
            print('The default column name is "SessionId", but you can specify otherwise by setting the `session_key` parameter of the model.')
            sys.exit(1)
        if args.item_key not in header:
            print('ERROR. The colmn specified for item IDs "{}" is not in the data file ({})'.format(args.item_key, fname))
            print('The default column name is "ItemId", but you can specify otherwise by setting the `item_key` parameter of the model.')
            sys.exit(1)
        if args.time_key not in header:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(args.time_key, fname))
            print('The default column name is "Time", but you can specify otherwise by setting the `time_key` parameter of the model.')
            sys.exit(1)
        if args.user_key not in header:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(args.user_key, fname))
            print('The default column name is "userId", but you can specify otherwise by setting the `user_key` parameter of the model.')
            sys.exit(1)
        if args.rel_int_key not in header:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(args.rel_int_key, fname))
            print('The default column name is "relational_interval", but you can specify otherwise by setting the `rel_int_key` parameter of the model.')
            sys.exit(1)
        print('Loading data from CSV file: {}'.format(fname))
        data = pd.read_csv(fname, sep=',', usecols=[args.user_key, args.session_key, args.item_key, args.time_key, args.rel_int_key], dtype={args.user_key:'int32', args.session_key:'str', args.item_key:'int32', args.time_key: 'int64'}, converters={'relational_interval':literal_eval})
    return data # returns df of dataset csv with SessionId, ItemId, Time cols

# exit if multiple of these parameters are provided
if (args.parameter_string is not None) + (args.parameter_file is not None) + (args.load_model) != 1:
    print('ERROR. Exactly one of the following parameters must be provided: --parameter_string, --parameter_file, --load_model')
    sys.exit(1)

# load dataset
data = load_data(args.path, args)

# set up ex2vec
if args.combination:
    ex2vec_param_str = convert_to_param_str(args.base_path + 'optim/best_params_ex2vec.json')
    ex2vec_config = OrderedDict([x.split('=') for x in ex2vec_param_str.split(',') if "=" in x])

    n_users = data[args.user_key].nunique()
    n_items = data[args.item_key].nunique()
    config = config = {
        "alias": 'ex2vec_baseline_finaltrain_DEL',
        "num_epoch": int(ex2vec_config['num_epoch']),
        "batch_size": int(ex2vec_config['batch_size']),
        "optimizer": 'adam',
        "lr": float(ex2vec_config['learning_rate']),
        "rmsprop_alpha": float(ex2vec_config['rmsprop_alpha']),
        "momentum": float(ex2vec_config['momentum']),
        "n_users": n_users,
        "n_items": n_items,
        "latent_dim": 64,
        "num_negative": 0,
        "l2_regularization": float(ex2vec_config['l2_regularization']),
        "use_cuda": True,
        "device_id": 0,
        "pretrain": True,
        "pretrain_dir": args.ex2vec_path,
        "model_dir": args.base_path + "models/{}_Epoch{}_f1{:.4f}.pt",
        "chckpt_dir": args.base_path + "chckpts/{}_Epoch{}_f1{:.4f}.pt",
    }

    ex2vec = Ex2VecEngine(config)
else:
    ex2vec = None

gru4rec_params = None
if args.load_model:
    print('Loading trained model from file: {} (to device "{}")'.format(args.path, args.device))
    gru = GRU4Rec.loadmodel(args.path, device=args.device)
else: # new model will be created and trained
    if args.parameter_file: # load training params from file
        """
        param_file_path = os.path.abspath(args.parameter_file) # get absolute path of parameter file
        param_dir, param_file = os.path.split(param_file_path) # split path into directory and file name
        spec = importlib.util.spec_from_file_location(param_file.split('.py')[0], os.path.abspath(args.parameter_file)) #where to find python file and how to load it
        params = importlib.util.module_from_spec(spec) # create empty module based on spec
        spec.loader.exec_module(params) # executes the paramfile .py
        gru4rec_params = params.gru4rec_params # accesses the stored params
        print('Loaded parameters from file: {}'.format(param_file_path))
        """
        param_str = convert_to_param_str(args.parameter_file)
        gru4rec_params = OrderedDict([x.split('=') for x in param_str.split(',')])
    if args.parameter_string: # if parameter string is provided, parse it and create an ordered dict of params -> load training params from string directly
        gru4rec_params = OrderedDict([x.split('=') for x in args.parameter_string.split(',')])
    print('Creating GRU4Rec model on device "{}"'.format(args.device))
    gru = GRU4Rec(device=args.device) # new GRU4Rec class instance saved onto specified device
    gru.set_params(**gru4rec_params) # set training params from file or string
    print('Loading training data...')
    print('Started training')
    t0 = time.time() # record current start time
    gru.fit(data, sample_cache_max_size=args.sample_store_size, item_key=args.item_key, session_key=args.session_key, time_key=args.time_key, combination=args.combination, ex2vec=ex2vec, alpha=args.alpha, save_path=args.base_path+'GRU4Rec_Fork/chckpts/GRU4Rec_chckpts_Epoch_{}_loss{:.6f}.pt' if not args.optim else None) # do not store checkpoints during hyperparameter tuning
    t1 = time.time() # record current end time
    print('Total training time: {:.2f}s'.format(t1 - t0))
    
if args.test is not None:
    # check what user has indicated to be the primary metric to be evaluated
    if args.primary_metric.lower() == 'recall':
        pm_index = 0
    elif args.primary_metric.lower() == 'mrr':
        pm_index = 1
    else:
        raise RuntimeError('Invalid value `{}` for `primary_metric` parameter'.format(args.primary_metric))
    for test_file in args.test:
        print('Loading test data...')
        test_data = load_data(test_file, args)
        print('Starting evaluation (cut-off={}, using {} mode for tiebreaking)'.format(args.measure, args.eval_type))
        t0 = time.time() # start time for eval
        # evaluate model on test data with specified batch size
        print("Eval batch size: ", gru4rec_params.get('batch_size'))
        res = evaluation.batch_eval(gru=gru, test_data=test_data, cutoff=args.measure, batch_size=int(gru4rec_params.get('batch_size')), mode=args.eval_type)
        t1 = time.time() # end time for eval
        print('Evaluation took {:.2f}s'.format(t1 - t0))

        metric_str = ''
        for i, c in enumerate(args.measure):
            print('Recall@{}: {:.6f} MRR@{}: {:.6f}'.format(c, res[0][c], c, res[1][c]))
            metric_str += f"recall@{c}={res[0][c]}, mrr@{c}={res[1][c]}"
            if i != (len(args.measure) - 1):
                metric_str += ", "

        if args.log_primary_metric:
            print('PRIMARY METRIC: {}'.format([x for x in res[pm_index].values()][0]))

    # save trained model
    if args.save_model is not None:
        gru.savemodel(path=args.save_model, param_str=args.parameter_string, metric_str=metric_str, log_path = args.base_path + 'results/best_models.csv', is_checkpoint=False)
