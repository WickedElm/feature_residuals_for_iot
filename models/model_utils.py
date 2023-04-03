import glob
import ipdb
import random
import os

def load_saved_model(root_dir='.', search_path='.', model_name='last.ckpt', model_class=None, method='random'):
    candidate_checkpoints = glob.glob(f'{root_dir}/{search_path}/**/{model_name}', recursive=True)

    if len(candidate_checkpoints) == 0:
        return None

    chosen_checkpoint_idx = random.randint(0, len(candidate_checkpoints) - 1)

    model = model_class.load_from_checkpoint(candidate_checkpoints[chosen_checkpoint_idx])
    return model, candidate_checkpoints[chosen_checkpoint_idx]
