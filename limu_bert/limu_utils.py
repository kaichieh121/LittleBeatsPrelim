import json
from typing import NamedTuple
import numpy as np

class PretrainModelConfig(NamedTuple):
    "Configuration for BERT model"
    hidden: int = 0  # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = 0  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    feature_num: int = 0  # Factorized embedding parameterization

    n_layers: int = 0  # Numher of Hidden Layers
    n_heads: int = 0  # Numher of Heads in Multi-Headed Attention Layers
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    seq_len: int = 0  # Maximum Length for Positional Embeddings
    emb_norm: bool = True

    @classmethod
    def from_json(cls, js):
        return cls(**js)


class ClassifierModelConfig(NamedTuple):
    "Configuration for classifier model"
    seq_len: int = 0
    input: int = 0

    num_rnn: int = 0
    num_layers: int = 0
    rnn_io: list = []

    num_cnn: int = 0
    conv_io: list = []
    pool: list = []
    flat_num: int = 0

    num_attn: int = 0
    num_head: int = 0
    atten_hidden: int = 0

    num_linear: int = 0
    linear_io: list = []

    activ: bool = False
    dropout: bool = False

    @classmethod
    def from_json(cls, js):
        return cls(**js)


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

'''
usage example
model_cfg = load_model_config(target='pretain_base', prefix='base', version='v1')
'''
def load_model_config(target, prefix, version, path_bert='/home/kcchang3/workplace/LittleBeatsPrelim/limu_bert/config/limu_bert.json', path_classifier='/home/kcchang3/workplace/LittleBeatsPrelim/limu_bert/config/classifier.json'):
    if "bert" not in target: # pretrain or pure classifier
        if "pretrain" in target:
            model_config_all = json.load(open(path_bert, "r"))
        else:
            model_config_all = json.load(open(path_classifier, "r"))
        name = prefix + "_" + version
        if name in model_config_all:
            if "pretrain" in target:
                return PretrainModelConfig.from_json(model_config_all[name])
            else:
                return ClassifierModelConfig.from_json(model_config_all[name])
        else:
            return None
    else: # pretrain + classifier for fine-tune
        model_config_bert = json.load(open(path_bert, "r"))
        model_config_classifier = json.load(open(path_classifier, "r"))
        prefixes = prefix.split('_')
        versions = version.split('_')
        bert_name = prefixes[0] + "_" + versions[0]
        classifier_name = prefixes[1] + "_" + versions[1]
        if bert_name in model_config_bert and classifier_name in model_config_classifier:
            return [PretrainModelConfig.from_json(model_config_bert[bert_name])
                , ClassifierModelConfig.from_json(model_config_classifier[classifier_name])]
        else:
            return None