from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

def get_scheduler(optimizer, config):
    name = config.get('name', 'cosine')
    if name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=config.get('epochs', 100))
    elif name == 'multistep':
        return MultiStepLR(optimizer, milestones=config.get('milestones', [50, 80]), gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
