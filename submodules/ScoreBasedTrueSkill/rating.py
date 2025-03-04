"""
Please ref to the original code here:
https://github.com/saulabs/trueskill/blob/master/lib/saulabs/trueskill/rating.rb
"""
from submodules.ScoreBasedTrueSkill import Gauss


# trueskill/lib/saulabs/trueskill/rating.rb 
class Rating(Gauss.Distribution):

    def __init__(self, mean=25.0, deviation=25.0/3, tau=25/300.0,  activity=1.0):
        super().__init__(mean = mean, deviation = deviation)
        self.activity = activity
        self.tau = tau
        
    @property
    def tau_squared(self):
        return self.tau ** 2