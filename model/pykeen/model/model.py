import re
import numpy as np
import pandas as pd
from utils import *
from pykeen.pipeline import pipeline
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.triples.triples_factory import _map_triples_elements_to_ids
from pykeen.triples.triples_factory import EntityMapping, RelationMapping

class Model():
    def __init__(self, train, valid, test, params):
        self.train = train
        self.valid = valid
        self.test = test
        self.model = params.model
        self.training_kwargs = params.training_kwargs
        self.model_kwargs = params.model_kwargs
        self.stopper = params.stopper
        self.stopper_kwargs = params.stopper_kwargs
        self.evaluation_kwargs = params.evaluation_kwargs
        self.optimizer_kwargs = params.optimizer_kwargs
        self.device = params.device

    def forward(self):
        result = pipeline(
            model=self.model,
            training=self.train,
            validation=self.valid,
            testing=self.test,
            training_kwargs=self.training_kwargs,  # 30
            model_kwargs=self.model_kwargs,
            stopper=self.stopper,
            stopper_kwargs=self.stopper_kwargs,
            evaluation_kwargs=self.evaluation_kwargs,
            optimizer_kwargs=self.optimizer_kwargs,
            device=self.device,
        )
        return result