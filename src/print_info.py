import torch

def print_info(save_dir):
    print(f'printing info saved in: {save_dir}')
    info = torch.load(save_dir + 'info.pt')
    acc = info['results']['performance']['acc']
    conf = info['results']['performance']['conf']
    print(f'accuracy: {acc}')
    print(f'confusion matrix: {conf}')
