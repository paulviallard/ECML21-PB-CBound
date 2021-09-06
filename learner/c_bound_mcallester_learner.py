import math
import torch

from core.cocob_optim import COCOB
from learner.gradient_descent_learner import GradientDescentLearner

###############################################################################


class CBoundMcAllesterLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, epoch=1, batch_size=None, delta=0.05, t=100
    ):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size)
        self.t = t
        self._optim = None
        self.delta = delta

    def __c_bound(self, r, d):
        """
        Compute the C-Bound (The "third form" in [1])

        Parameters
        ----------
        r: tensor
            The risk
        d: tensor
            The disagreement
        """
        r = torch.min(torch.tensor(0.5).to(r.device), r)
        d = torch.max(torch.tensor(0.0).to(d.device), d)
        cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
        if(torch.isnan(cb) or torch.isinf(cb)):
            cb = torch.tensor(1.0, requires_grad=True)
        return cb

    def __risk_bound(self, rS, kl, m):
        """
        Compute the PAC-Bayesian bound on the risk

        Parameters
        ----------
        rS: tensor
            The empirical risk
        kl: tensor
            The KL divergence
        m: float
            The number of data
        """
        b = (1.0/(2.0*m))*(kl+math.log(
            (2.0*math.sqrt(m))/(0.5*self.delta)))
        b = rS + torch.sqrt(b)
        return b

    def __disagreement_bound(self, dS, kl, m):
        """
        Compute the PAC-Bayesian bound on the disagreement

        Parameters
        ----------
        dS: tensor
            The empirical disagreement
        kl: tensor
            The KL divergence
        m: float
            The number of data
        """
        b = (1.0/(2.0*m))*(2.0*kl+math.log(
            (2.0*math.sqrt(m))/(0.5*self.delta)))
        b = dS - torch.sqrt(b)
        return b

    def __log_barrier(self, x):
        """
        Compute the log-barrier extension of [2]

        Parameters
        ----------
        x: tensor
            The constraint to optimize
        """
        assert isinstance(x, torch.Tensor) and len(x.shape) == 0
        # We use the
        if(x <= -1.0/(self.t**2.0)):
            return -(1.0/self.t)*torch.log(-x)
        else:
            return self.t*x - (1.0/self.t)*math.log(1/(self.t**2.0))+(1/self.t)

    def _optimize(self, batch):
        """
        Optimize the PAC-Bayesian bound of [3]

        Parameters
        ----------
        batch: dict
            The examples of the batch
        """
        if(self._optim is None):
            self._optim = COCOB(self.mv_diff.parameters())

        # We compute the prediction of the majority vote
        self.mv_diff(batch)
        pred = self.mv_diff.pred
        kl = self.mv_diff.kl

        assert "y" in batch and isinstance(batch["y"], torch.Tensor)
        y = batch["y"]
        y_unique = torch.sort(torch.unique(y))[0]
        assert y_unique[0].item() == -1 and y_unique[1].item() == +1

        assert len(y.shape) == 2 and len(pred.shape) == 2
        assert (pred.shape[0] == y.shape[0] and pred.shape[1] == y.shape[1]
                and y.shape[1] == 1)
        assert len(kl.shape) == 0

        # We compute the empirical joint error; empirical disagreement
        # and the bounds
        rS = torch.mean((0.5*(1.0-y*pred)))
        dS = torch.mean(0.5*(1.0-(pred**2.0)))
        m = batch["m"]
        r = self.__risk_bound(rS, kl, m)
        d = self.__disagreement_bound(dS, kl, m)

        # We optimize the PAC-Bayesian C-Bound of [3]
        self._loss = self.__c_bound(r, d)
        self._loss += self.__log_barrier(r-0.5)

        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

        self._log["c-bound"] = self.__c_bound(r, d)
        self._log["0-1 loss"] = 0.5*(1.0-torch.mean(torch.sign(y*pred)))

###############################################################################

# References:
# [1] Risk Bounds for the Majority Vote:
#     From a PAC-Bayesian Analysis to a Learning Algorithm
#     Pascal Germain, Alexandre Lacasse, Francois Laviolette,
#     Mario Marchand, Jean-Francis Roy, 2015
# [2] Constrained Deep Networks: Lagrangian Optimization
#     via Log-Barrier Extensions
#     Hoel Kervadec, Jose Dolz, Jing Yuan, Christian Desrosiers,
#     Eric Granger, Ismail Ben Ayed, 2019
# [3] A Column Generation Bound Minimization Approach with
#     PAC-Bayesian Generalization Guarantees
#     Jean-Francis Roy, Mario Marchand, François Laviolette, 2016
