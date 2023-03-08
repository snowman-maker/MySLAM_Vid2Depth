import configer

cfg = configer.load_config('./config.yaml')

name, dataset_dir, data_dir, seq_len = configer.gen_dataset(cfg)