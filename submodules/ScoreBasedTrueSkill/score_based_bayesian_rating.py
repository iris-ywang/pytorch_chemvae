"""
Please ref the use and method description here:
https://github.com/saulabs/trueskill/blob/master/lib/saulabs/trueskill/score_based_bayesian_rating.rb;
and this paper:
https://www.researchgate.net/publication/260639614_Score-Based_Bayesian_Skill_Learning
"""

from submodules.ScoreBasedTrueSkill.rating import Rating

class ScoreBasedBayesianRating:
    def __init__(self, score_teams_keys, score_teams_values, beta = 25/6, options = {}):
        self.teams = score_teams_keys
        self.scores = score_teams_values
        if not len(self.teams) == 2:
            raise "teams.size should be 2: this implementation of the score based bayesian rating only works for two teams" 
      
        opt = {
              "beta": beta,
              "skills_additive" : True
              }
        opts = {**opt, **options}
        
        self.beta = opts['beta']
        self.beta_squared = self.beta ** 2
        self.skills_additive = opts['skills_additive']
        self.gamma = options["gamma"] if 'gamma' in options else 0.1
        self.gamma_squared = self.gamma ** 2
        
        #self.teams = teams

      

    def update_skills(self): #oK
        #game can be 1vs1, 1vs2, 1vs3 or 2vs2
        #

        #team1 vs team2
        # if self.skills_additive = true: no averaging of skills and variance
        # otherwise: mean and skill_deviation averaged over team sizes
        n_team_1    = 1 if self.skills_additive else float(len(self.teams[0]))
        n_team_2    = 1 if self.skills_additive else float(len(self.teams[1]))
        

        n_all       = float(len(self.teams[0])) + float(len(self.teams[1]))
        var_team_1  = sum([item.variance for item in self.teams[0]])
        var_team_2  = sum([item.variance for item in self.teams[1]])
        mean_team_1  = sum([item.mean for item in self.teams[0]])
        mean_team_2  = sum([item.mean for item in self.teams[1]])
        

        for i in range(len(self.teams[0])):
            rating = self.teams[0][i]
            precision = 1.0 / rating.variance + 1.0/ ( n_all * self.beta_squared + 2.0 * self.gamma_squared + var_team_2 / n_team_2 + var_team_1 / n_team_1 - rating.variance / n_team_1)
            precision_mean = rating.mean / rating.variance + (self.scores[0] - self.scores[1] + n_team_1 * (mean_team_2 / n_team_2 - mean_team_1 / n_team_1 + rating.mean / n_team_1)) / ( n_all * self.beta_squared + 2.0 * self.gamma_squared + var_team_2 / n_team_2 + var_team_1 / n_team_1 - rating.variance / n_team_1)
            partial_updated_precision = rating.precision + rating.activity*( precision - rating.precision)
            partial_updated_precision_mean =  rating.precision_mean + rating.activity * (precision_mean - rating.precision_mean)
            self.teams[0][i] = Rating(partial_updated_precision_mean / partial_updated_precision, ( 1.0 / partial_updated_precision + rating.tau_squared)**0.5, rating.activity, rating.tau)
            

        for i in range(len(self.teams[1])):
            rating = self.teams[1][i]
            precision = 1.0 / rating.variance + 1.0 / (n_all*self.beta_squared + 2.0 * self.gamma_squared + var_team_1 / n_team_1 + var_team_2 / n_team_2 - rating.variance / n_team_2)
            precision_mean = rating.mean / rating.variance + (self.scores[1] - self.scores[0] + n_team_2 * (mean_team_1 / n_team_1 - mean_team_2 / n_team_2 + rating.mean / n_team_2)) / ( n_all * self.beta_squared + 2.0 * self.gamma_squared + var_team_1 / n_team_1 + var_team_2/n_team_2 - rating.variance / n_team_2)
            partial_updated_precision = rating.precision + rating.activity*( precision - rating.precision)
            partial_updated_precision_mean =  rating.precision_mean + rating.activity * (precision_mean - rating.precision_mean)
            self.teams[1][i] = Rating(partial_updated_precision_mean / partial_updated_precision, (1.0 / partial_updated_precision + rating.tau_squared)**0.5, rating.activity, rating.tau)
        



