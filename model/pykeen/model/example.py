import re
import torch
import numpy as np
import pandas as pd
from utils import *
from predict import *
from pykeen.pipeline import pipeline
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.triples.triples_factory import _map_triples_elements_to_ids
from pykeen.triples.triples_factory import EntityMapping, RelationMapping

result = DrugMap('druga', 'drug').query()
print(result)
print()
result = DrugMap('proteina', 'protein').query()
print(result)
print()
result = DrugMap('genea', 'gene').query()
print(result)