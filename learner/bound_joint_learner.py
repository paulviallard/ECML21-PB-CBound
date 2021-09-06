import math
import torch

from core.irprop_optim import iRProp
from learner.gradient_descent_learner import GradientDescentLearner

###############################################################################


class BoundJointLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, epoch=1, batch_size=None, delta=0.05, t=100
    ):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size)
        self.max_iterations = 1000

        self._old_loss = float("inf")
        self._eps = 10**-9

        self._optim = None
        self.delta = delta

    def __lambda(self, eS, kl, m):
        """
        Compute the lambda parameter of the bound used in [1]

        Parameters
        ----------
        eS: tensor
            The empirical joint error
        kl: tensor
            The KL divergence
        m: float
            The number of data
        """
        lambda_ = torch.sqrt(
            (2.0*m*eS) / (2.0*kl+math.log(2.0*math.sqrt(m)/self.delta))+1)
        lambda_ = 2.0 / (lambda_ + 1)
        lambda_ = lambda_.data
        return lambda_

    def __bound(self, eS, lambda_, kl, m):
        """
        Compute the bound used in [1]

        Parameters
        ----------
        eS: tensor
            The empirical joint error
        lambda_: tensor
            The lambda parameter
        kl: tensor
            The KL divergence
        m: float
            The number of data
        """
        bound = eS/(1.0-lambda_/2.0)
        bound = bound + ((2.0*kl+math.log(2.0*math.sqrt(m)/self.delta))
                         / (lambda_*(1.0-lambda_/2.0)*m))
        bound = 4*bound
        return bound

    def _optimize(self, batch):
        """
        Optimize the bound used in [1]

        Parameters
        ----------
        eS: tensor
            The empirical joint error
        lambda_: tensor
            The lambda parameter
        kl: tensor
            The KL divergence
        m: float
            The number of data
        """
        self._optim = iRProp(self.mv_diff.parameters())

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

        eS = torch.mean((0.5*(1.0-y*pred))**2.0)
        m = batch["m"]

        assert pred.shape[0] == m

        lambda_ = self.__lambda(eS, kl, m)
        self._loss = self.__bound(eS, lambda_, kl, m)


        # For a given number of iterations ...
        t = 1
        t_best = 0
        loss_list = [self._loss]
        post_list = [self.mv_diff.post.data]
        while t < self.max_iterations:

            # If the loss does not improve for 10 iterations, we stop
            if t-t_best > 10:
                break

            # We backpropagate
            loss_list.append(self._loss)
            self._optim.zero_grad()
            self._loss.backward()
            self._optim.step(self._loss)
            post_list.append(self.mv_diff.post.data)

            # We compute the empirical joint error; KL divergence and the bound
            self.mv_diff(batch)
            pred = self.mv_diff.pred
            eS = torch.mean((0.5*(1.0-y*pred))**2.0)
            kl = self.mv_diff.kl
            self._loss = self.__bound(eS, lambda_, kl, m)

            # We update the best iteration
            if(loss_list[t] < loss_list[t_best]):
                t_best = t

            t += 1
            self.epoch_ += 1

        self.mv_diff.post.data = post_list[t_best]

        # We compute the empirical joint error; KL divergence and the bound
        self.mv_diff(batch)
        pred = self.mv_diff.pred
        eS = torch.mean((0.5*(1.0-y*pred))**2.0)
        kl = self.mv_diff.kl
        self._loss = self.__bound(eS, lambda_, kl, m)

        # If the loss does not improve, we stop the optimization
        if(torch.abs(self._old_loss-self._loss) <= self._eps):
            self.epoch_ = self.epoch-1
        elif(self._loss > self._old_loss):
            self.mv_diff.post.data = self._old_post
            self.epoch_ = self.epoch-1

        self._old_post = self.mv_diff.post.data
        self._old_loss = self._loss

        self._log["bound"] = self._loss
        self._log["0-1 loss"] = 0.5*(1.0-torch.mean(torch.sign(y*pred)))

###############################################################################

# References:
# [1] Second Order PAC-Bayesian Bounds for the Weighted Majority Vote
#     Andrés R. Masegosa, Stephan S. Lorenzen, Christian Igel, Yevgeny Seldin,
#     2020
