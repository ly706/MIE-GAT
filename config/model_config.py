from collections import OrderedDict
config = OrderedDict()
config['dataset_name'] = 'LIDP' # 'LIDP' or 'LIDC-IDRI'
config['slice_emb_size'] = 64
# GAT
config['feat_in'] = 64
config['hidden'] = 64
config['nclass'] = 2
config['gat_dropout'] = 0.2
config['nb_heads'] = 4  # multi-head attention
config['alpha'] = 0.2  # Alpha for the leaky_relu
config['dim_num'] = 3
config['imgNum'] = 32
config['struct_fea'] = 9

