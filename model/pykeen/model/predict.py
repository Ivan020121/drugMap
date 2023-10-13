import re
import torch
import numpy as np
import pandas as pd
from utils import *
from pykeen.models import predict
from pykeen.pipeline import pipeline
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.triples.triples_factory import _map_triples_elements_to_ids
from pykeen.triples.triples_factory import EntityMapping, RelationMapping


class DrugMap():
    def __init__(self, query_name, query_type):
        self.query_name = query_name
        self.query_type = query_type
        self.protein = np.load('../data/protein.npy').tolist()
        self.drug = np.load('../data/drug.npy').tolist()
        self.gene = np.load('../data/gene.npy').tolist()
        self.disease = np.load('../data/disease.npy').tolist()
        self.model = torch.load('./result/trained_model.pkl')
        self.triple_factory = load_triple_factory(10, 10)

    def get_relation(self, head_type):
        """Get relation.

        :param head_type: head_type(str)
        :return: {relation_name: tail_set...}
        """
        if head_type == 'drug':
            return {'ddi': [self.drug, 'drug'], 'dpi': [self.protein, 'protein'], 'dgi': [self.gene, 'gene']}
        elif head_type == 'protein':
            return {'ppi': [self.protein, 'protein'], 'psi': [self.disease, 'disease'], 'pgi': [self.gene, 'gene']}
        elif head_type == 'gene':
            return {'gdi': [self.drug, 'drug'], 'ggi': [self.gene, 'gene'], 'gpi': [self.protein, 'protein'],
                    'gsi': [self.disease, 'disease']}
        else:
            return {'ssi': [self.disease, 'disease']}

    def get_interaction(self, head, relation, tail_set, tail_type):
        """Get interaction.

        :param head: head_name(str)
        :param relation: relation_name(str)
        :param tail_set: tail_set(list)
        :param tail_type: tail_type(str)
        :return: [(h, r, t, t_type), ...]
        """
        interaction = []
        prediction = predict.get_tail_prediction_df(model=self.model, head_label=head, relation_label=relation,
                                                    triples_factory=self.triple_factory, remove_known=True)
        tail_label = prediction.iloc[0:10, 1].tolist()
        tail_label = list(set(tail_set) & set(tail_label))
        for tail in tail_label:
            interaction.append((head, relation, tail, tail_type))
            # interaction.append((head, relation, tail, tail_type) for tail in tail_label)
        return interaction

    def get_layer(self, head_set):
        """Get layer.

        :param head_set: (head_num, 2)->(head_name, head_type)
        :return: {head:[(h, r, t, t_type), ...], ...}
        """
        layer = {}
        result_model = self.model
        for head, type in head_set:
            layer[head] = []
            relation = self.get_relation(type)
            for k, v in relation.items():
                # k: relation_name, v: tail([tail_set, tail_type])
                interaction = self.get_interaction(head, k, v[0], v[1])
                for i in interaction:
                    layer[head].append(i)
        return layer

    def query(self):
        """Get query result.

        :return: {0: ['a'],
        1: {'a': [('a', 'dpi', 'b', 'protein'), ('a', 'dpi', 'c', 'protein'), ('a', 'dgi', 'd', 'gene')]},
        2: {'b': [('b', 'ppi', 'e', 'protein'), ('b', 'pgi', 'f', 'gene'), ('b', 'psi', 'g', 'disease')], 'c': [('c', 'ppi', 'e', 'protein'), ('c', 'pgi', 'f', 'gene'), ('c', 'psi', 'g', 'disease')], 'd': [('d', 'gpi', 'e', 'protein'), ('d', 'ggi', 'f', 'gene'), ('d', 'gsi', 'g', 'disease')]}}
        """
        query_result = {}
        query_layer_1 = [(self.query_name, self.query_type)]
        layer_1 = self.get_layer(query_layer_1)
        query_result[0] = [self.query_name]
        query_result[1] = layer_1
        if self.query_type == 'drug':
            query_layer_2 = []
            for key, value in layer_1.items():
                for v in value:
                    query_layer_2.append((v[2], v[3]))
            layer_2 = self.get_layer(query_layer_2)
            query_result[2] = layer_2
        return query_result
