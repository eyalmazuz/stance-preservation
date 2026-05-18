import torch
import numpy as np
from src.models.emd import EMDScorer

class MockScorer(EMDScorer):
    def __init__(self, score_method):
        self.score_method = score_method
        self.use_dist_topic_score = False
        self.use_weighted_emd = False
        self.debug = False
        self.use_topic_filtering = False
        self.use_soft_topic_filtering = False
        self.entropy_threshold = float('inf')
        self.stance_value = {"Against": -1, "Neutral": 0, "Favor": 1}
        self.canonical_labels = ["Against", "Neutral", "Favor"]
        
    def get_matching_pairs(self, h, r):
        return [("h1", "r1", 1.0)]
        
    def get_topic(self, ft, s):
        return "topic"
        
    def get_stance(self, s, t):
        return torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)

scorer = MockScorer("euclidean")
print("euclidean score:", scorer.score("hyp", "ref"))

scorer = MockScorer("itakura")
print("itakura score:", scorer.score("hyp", "ref"))

