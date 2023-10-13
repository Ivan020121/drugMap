import re
import torch
import numpy as np
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.triples.triples_factory import _map_triples_elements_to_ids
from pykeen.triples.triples_factory import EntityMapping, RelationMapping

INVERSE_SUFFIX = '_inverse'


class ProDataset():
    def __init__(self, params):
        self.train_path = params.train_path
        self.valid_path = params.valid_path
        self.test_path = params.test_path
        self.num_entities = params.num_entities
        self.num_relations = params.num_relations

    def get_triples_and_relations(self):
        """Get triples and relations from dataset.

        :param self: object
        :returns triples, train_triples, valid_triples, test_triples, relation_set
        """
        names = ['head', 'relation', 'tail']
        train = pd.read_csv(self.train_path, sep='\t', header=None, names=names)
        valid = pd.read_csv(self.valid_path, sep='\t', header=None, names=names)
        test = pd.read_csv(self.test_path, sep='\t', header=None, names=names)
        df = train.append(valid).append(test).reset_index(drop=True)
        return df.values, train.values, valid.values, test.values, set(df['relation'])

    def create_entity_mapping(self, triples) -> EntityMapping:
        """Create mapping from entity labels to IDs.

        :param triples: shape: (n, 3), dtype: str
        :return entity_to_id
        """
        # Split triples
        heads, tails = triples[:, 0], triples[:, 2]
        # Sorting ensures consistent results when the triples are permuted
        entity_labels = sorted(set(heads).union(tails))
        # Create mapping
        return {
            str(label): i
            for (i, label) in enumerate(entity_labels)
        }

    def create_relation_mapping(self, relations: set) -> RelationMapping:
        """Create mapping from relation labels to IDs.

        :param relations: set
        :return relation_to_id
        """
        # Sorting ensures consistent results when the triples are permuted
        relation_labels = sorted(
            set(relations),
            key=lambda x: (re.sub(f'{INVERSE_SUFFIX}$', '', x), x.endswith(f'{INVERSE_SUFFIX}')),
        )
        # Create mapping
        return {
            str(label): i
            for (i, label) in enumerate(relation_labels)
        }

    def get_mapped_triples(self, triples, entity_to_id, relation_to_id):
        """Get ID-based triples either directly, or from a factory.

        :param triples: shape: (n, 3), dtype: str
        :param entity_to_id: The mapping from entities' labels to their indices.
        :param relation_to_id: The mapping from relations' labels to their indices.
        :return triple_to_id
        """
        return _map_triples_elements_to_ids(triples, entity_to_id, relation_to_id)

    def get_triples_factory(self, mapped_triples, entity_to_id, relation_to_id):
        """Get TriplesFactory.

        :param mapped_triples:
        :param entity_to_id:
        :param relation_to_id:
        :return: TriplesFactory
        """
        triples_factory = TriplesFactory(mapped_triples=mapped_triples, entity_to_id=entity_to_id,
                                         relation_to_id=relation_to_id)
        triples_factory._num_entities = self.num_entities
        triples_factory._num_relations = self.num_relations
        return triples_factory

    def load_data(self):
        """Load train, valid, test data.

        :return: train_triples_factory, valid_triples_factory, test_triples_factory
        """
        triples, train_triples, test_triples, valid_triples, relations = self.get_triples_and_relations()
        entity_to_id = self.create_entity_mapping(triples)
        relation_to_id = self.create_relation_mapping(relations)
        train_mapped_triples = self.get_mapped_triples(train_triples, entity_to_id, relation_to_id)
        valid_mapped_triples = self.get_mapped_triples(valid_triples, entity_to_id, relation_to_id)
        test_mapped_triples = self.get_mapped_triples(test_triples, entity_to_id, relation_to_id)
        train = self.get_triples_factory(train_mapped_triples, entity_to_id, relation_to_id)
        valid = self.get_triples_factory(valid_mapped_triples, entity_to_id, relation_to_id)
        test = self.get_triples_factory(test_mapped_triples, entity_to_id, relation_to_id)
        return train, valid, test


def list_operator(list_path, ls=None):
    if bool(ls):
        ls = np.array(ls)
        np.save(list_path, ls)
    else:
        ls = np.load(list_path).tolist()
        return ls


def load_triple_factory(num_entities, num_relations):
    tsv = pd.read_csv('./result/training_triples/entity_to_id.tsv', sep='\t')
    entity_to_id = dict(zip(tsv['label'], tsv['id']))
    tsv = pd.read_csv('./result/training_triples/relation_to_id.tsv', sep='\t')
    relation_to_id = dict(zip(tsv['label'], tsv['id']))
    tsv = pd.read_csv('./result/training_triples/numeric_triples.tsv', sep='\t')
    mapped_triples = torch.LongTensor([tuple(x) for x in tsv.values])
    triples_factory = TriplesFactory(mapped_triples=mapped_triples, entity_to_id=entity_to_id,
                                     relation_to_id=relation_to_id)
    triples_factory._num_entities = num_entities
    triples_factory._num_relations = num_relations
    return triples_factory