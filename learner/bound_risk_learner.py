import torch
import math

from core.cocob_optim import COCOB
from core.kl_inv import klInvFunction
from learner.gradient_descent_learner import GradientDescentLearner

###############################################################################


class BoundRiskLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, epoch=1, batch_size=None, delta=0.05, t=100
    ):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size)
        self.t = t
        self._optim = None
        self.delta = delta

    def __bound(self, kl, m):
        """
        Compute the PAC-Bayesian bound

        Parameters
        ----------
        kl: tensor
            The KL divergence
        m: float
            The number of data
        """
        return (1.0/m)*(kl+math.log((2.0*math.sqrt(m))/self.delta))

    def __risk_bound(self, rS, kl, m):
        """
        Compute the (Seeger's) PAC-Bayesian bound for the risk

        Parameters
        ----------
        rS: tensor
            The empirical risk
        kl: tensor
            The KL divergence
        m: float
            The number of data
        """
        kl_inv = klInvFunction.apply
        b = self.__bound(kl, m)
        b = kl_inv(rS, b, "MAX")
        return b

    def _optimize(self, batch):
        """
        Optimize the PAC-Bound 0 by gradient descent (see [1])

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

        # We compute the empirical risk and the bound
        rS = torch.mean((0.5*(1.0-y*pred)))
        m = batch["m"]
        r = self.__risk_bound(rS, kl, m)
        self._loss = 2.0*r

        # We backward
        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

        self._log["bound"] = self._loss
        self._log["0-1 loss"] = 0.5*(1.0-torch.mean(torch.sign(y*pred)))

###############################################################################

# References:
# [1] Risk bounds for the majority vote:
#     from a PAC-Bayesian analysis to a learning algorithm
#     Pascal Germain, Alexandre Lacasse, François Laviolette,
#     Mario Marchand, Jean-Francis Roy, 2015
