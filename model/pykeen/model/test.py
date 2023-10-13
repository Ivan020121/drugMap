import re
import argparse
import numpy as np
import pandas as pd
import torch
from utils import *
from model import *
from typing import List
from pykeen.models import predict
from pykeen.datasets import Nations
from pykeen.pipeline import pipeline
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.triples.triples_factory import _map_triples_elements_to_ids
from pykeen.triples.triples_factory import EntityMapping, RelationMapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser For Arguments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-train_path', dest="train_path", default='../data/train.tsv', help='Train dataset to use')
    parser.add_argument('-valid_path', dest="valid_path", default='../data/valid.tsv', help='Valid dataset to use')
    parser.add_argument('-test_path', dest="test_path", default='../data/test.tsv', help='Test dataset to use')
    parser.add_argument('-num_entities', dest="num_entities", default=10, help='Number of entities')
    parser.add_argument('-num_relations', dest="num_relations", default=10, help='Number of relations')

    parser.add_argument("-model", dest='model', default='TransE', help="Knowledge graph embedding model")
    parser.add_argument("-training_kwargs", dest='training_kwargs', default={'num_epochs':300}, help="Training kwargs")
    parser.add_argument("-model_kwargs", dest='model_kwargs', default={'embedding_dim':30}, help="Model kwargs")
    parser.add_argument("-stopper", dest='stopper', default='early', help="Early stoppers")
    parser.add_argument("-stopper_kwargs", dest='stopper_kwargs', default={'frequency':10, 'stopped':False, 'patience':1}, help="Stopper kwargs")
    parser.add_argument("-evaluation_kwargs", dest='evaluation_kwargs', default={'batch_size':32}, help="Evaluation kwargs")
    parser.add_argument("-optimizer_kwargs", dest='optimizer_kwargs', default={'lr':0.1}, help="Optimizer kwargs")
    parser.add_argument("-device", dest='device', default='cuda', help="Device")

    args = parser.parse_args()

    train, valid, test = ProDataset(args).load_data()
    result = Model(train, valid, test, args).forward()
    # print(result.losses)
    # print(result.metric_results.get_metric('mean_reciprocal_rank'))
    # print(predict.get_tail_prediction_df(model=result.model, head_label='druga', relation_label='dpi', remove_known=False, triples_factory=result.training))
    result.save_to_directory('result')
    # model = torch.load('./result/trained_model.pkl')
    # print(predict.get_tail_prediction_df(model=model, head_label='druga', relation_label='dpi', remove_known=False, triples_factory=train))