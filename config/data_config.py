import math
import os

class config:
    train_ratio = 1.0
    crop_size = 32
    slice_num = 32
    metric = 'auc'  # 可选'auc'和'acc'
    threshold=0.5
    cross_validation_num = 5
    struct_num = 7
    # struct_num = 9
    emb_size = 128

    gat_batch_size = 256

    # data path
    # LIDP
    path = '/media/data1/LC015Nodule'
    # LIDC
    # path = '/media/data1/newLIDCnodule'

    path_label = os.path.join(path, "feature.xlsx")

    # LIDP
    path_nodule = os.path.join(path, "alldata")
    # LIDC
    # path_nodule = os.path.join(path, "newalldata")

    path_split = [os.path.join(path, "split1.json"), os.path.join(path, "split2.json")]
    # LIDP
    sample_weight = 5
    sample_count_train = math.ceil(((136 + 26 * 5) * 4 * 32) * 1.0)
    sample_count_val = (136 + 26 * 5) * 4

    # # LIDC
    # sample_weight = 1
    # sample_count_train = 133 * 4 * 32 * 2
    # sample_count_val = 133 * 32


