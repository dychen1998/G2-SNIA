import os

from torch.utils.tensorboard import SummaryWriter

from default import visualizations_path


def make_writer(dataset_name: str, model_name: str):
    log_dir = os.path.join(visualizations_path, dataset_name, model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=120)
    return writer
