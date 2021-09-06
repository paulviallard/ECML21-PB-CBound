import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from voter.majority_vote import MajorityVote

###############################################################################


class MinCqLearner(BaseEstimator, ClassifierMixin):

    def __init__(self, majority_vote, mu):
        assert mu > 0 and mu <= 1
        self.mu = mu
        self.majority_vote = majority_vote
        self.mv = majority_vote

        assert isinstance(self.mv, MajorityVote)
        assert self.mv.fitted
        assert self.mv.complemented
        self.quasi_uniform = self.mv.quasi_uniform

    def get_params(self, deep=True):
        return {"mu": self.mu, "majority_vote": self.mv}

    def fit(self, X, y):
        """
        Run the algorithm MinCq (Program 1 in [1])

        Parameters
        ----------
        X: ndarray
            The inputs
        y: ndarray
            The labels
        """
        # X -> (size, nb_feature)
        # y -> (size, 1)
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(X.shape) == 2 and len(y.shape) == 2 and
                X.shape[0] == y.shape[0] and
                y.shape[1] == 1 and X.shape[0] > 0)
        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        if(not(self.quasi_uniform)):
            self.mv.normal_to_quasi_uniform()
        self.mv.switch_complemented()

        out = self.mv.output(X)

        # We define Program 1
        nb_voter = len(self.mv.post)

        M = (1.0/nb_voter)*(out.T@out)
        m = np.mean(y*out, axis=0)
        a = (1/nb_voter)*np.sum(M, axis=0)

        post_ = cp.Variable(shape=(nb_voter, 1))

        prob = cp.Problem(
            cp.Minimize(cp.quad_form(post_, M)-a.T@post_),
            [post_ >= 0.0, post_ <= 1.0/(nb_voter),
             2.0*(m.T@post_)-np.mean(m) == self.mu
             ])

        # We solve Program 1
        try:
            prob.solve(solver=cp.CVXOPT)
            post = post_.value
        except cp.error.SolverError:
            post = None

        if(post is None):
            post = np.array(self.mv.prior)
        post = 2.0*post-(1.0/nb_voter)
        self.mv.post = post

        self.mv.switch_complemented()
        if(not(self.quasi_uniform)):
            self.mv.quasi_uniform_to_normal()

        return self

    def predict(self, X):
        """
        Predict the label of the inputs

        X: ndarray
            The inputs
        """
        return self.mv.predict(X)

###############################################################################

# References:
# [1] From PAC-Bayes Bounds to Quadratic Programs for Majority Votes
#     François Laviolette, Mario Marchand, Jean-Francis Roy, 2011
