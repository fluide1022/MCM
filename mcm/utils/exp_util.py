import os.path

def build_exp_dir(cfg):
    exp_root = cfg['path']['experiment_root']
    # if os.path.exists(exp_root):
    #     new_path = exp_root + str(time.time())
    #     shutil.move(exp_root, new_path)
    #     print(f'exp root: {exp_root} already exists, moved to {new_path}')
    os.makedirs(exp_root, exist_ok=True)
    ckpt_root = os.path.join(exp_root, 'checkpoints')
    os.makedirs(ckpt_root, exist_ok=True)
    meta_root = os.path.join(exp_root, 'meta')
    os.makedirs(meta_root, exist_ok=True)
    cfg['path']['ckpt_root'] = ckpt_root
    cfg['path']['meta_root'] = meta_root
    return cfg
