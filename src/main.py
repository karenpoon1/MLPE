from src.models.M2PL import M2PL
from src.eval_model import eval_model
from src.print_info import print_info

# Variables
DATASET = 'b1'
SELECTED_PAPERS = [0]

SPLIT_CONFIG = 'default'
SPLIT_SEED = 1000

MODEL_DIM = 1
MODEL_CONFIG = 'default'
INIT_SEED = 1000
PLOT = True

# Folder name to save results
selected_papers_str = '_'.join(map(str, SELECTED_PAPERS))
save_dir = f'results/Dataset_{DATASET}__Paper{selected_papers_str}/SplitConfig_{SPLIT_CONFIG}__Random{SPLIT_SEED}/Model{MODEL_DIM}D2PL__ModelConfig_{MODEL_CONFIG}__InitRandom{INIT_SEED}/'

# Evaluate model
my_model = M2PL(MODEL_DIM)
info = eval_model(DATASET, SELECTED_PAPERS,
                 SPLIT_CONFIG, SPLIT_SEED,
                 my_model, MODEL_CONFIG, INIT_SEED,
                 PLOT, False)
print(info['results']['performance']['acc'])
print(info['results']['performance']['conf'])

# print_info(save_dir)
