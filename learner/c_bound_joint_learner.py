import cvxpy as cp
import math
import torch

from core.cocob_optim import COCOB
from learner.gradient_descent_learner import GradientDescentLearner

###############################################################################


class CBoundJointLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, epoch=1, batch_size=None, delta=0.05, t=100
    ):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size)
        self.t = t
        self._optim = None
        self.delta = delta

    def __c_bound(self, e, d):
        """
        Compute the C-Bound of PAC-Bound 2 (see page 820 of [1])

        Parameters
        ----------
        e: tensor
            The joint error
        d: tensor
            The disagreement
        """
        return (1.0-((1.0-(2.0*e+d))**2.0)/(1.0-2.0*d))

    def __log_barrier(self, x):
        """
        Compute the log-barrier extension of [2]

        Parameters
        ----------
        x: tensor
            The constraint to optimize
        """
        assert isinstance(x, torch.Tensor) and len(x.shape) == 0
        if(x <= -1.0/(self.t**2.0)):
            return -(1.0/self.t)*torch.log(-x)
        else:
            return self.t*x - (1.0/self.t)*math.log(1/(self.t**2.0))+(1/self.t)

    def __bound(self, kl, m, delta):
        """
        Compute the PAC-Bayesian bound of PAC-Bound 2 (see page 820 of [1])

        Parameters
        ----------
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        delta: float
            The confidence parameter of the bound
        """
        b = math.log((2.0*math.sqrt(m)+m)/delta)
        b = (1.0/m)*(2.0*kl+b)
        return b

    def __kl_tri(self, q1, q2, p1, p2):
        """
        Compute the KL divergence between two trinomials
        (see eq. (31) of [1])

        Parameters
        ----------
        q1: tensor
            The first parameter of the posterior trinomial distribution
        q2: tensor
            The second parameter of the posterior trinomial distribution
        p1: tensor
            The first parameter of the prior trinomial distribution
        p2: tensor
            The second parameter of the prior trinomial distribution
        """
        kl = torch.tensor(0.0).to(q1.device)
        if(q1 > 0):
            kl += q1*torch.log(q1/p1)
        if(q2 > 0):
            kl += q2*torch.log(q2/p2)
        if(q1+q2 < 1):
            kl += (1-q1-q2)*torch.log((1-q1-q2)/(1-p1-p2))
        return kl

    def __optimize_given_eS_dS(self, eS, dS, kl, m, tol=0.01):
        """
        Solve the inner maximization problem using the
        "Bisection method for quasiconvex optimization" of [3] (p 146)

        Parameters
        ----------
        eS: ndarray
            The empirical joint error
        dS: ndarray
            The empirical disagreement
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        tol: float, optional
            The tolerance parameter
        """
        u = 1.0
        l = 0.0
        bound = self.__bound(kl, m, self.delta).item()

        while(u-l > tol):
            t = (l+u)/2.0

            e = cp.Variable(shape=1, nonneg=True)
            d = cp.Variable(shape=1, nonneg=True)
            e_min = cp.atoms.affine.hstack.hstack([e, 0.25])
            prob = cp.Problem(
                cp.Minimize((1-(2*e+d))**2.0-t*(1-2*d)),
                [(cp.kl_div(eS, e)+cp.kl_div(dS, d)
                  + cp.kl_div((1-eS-dS), 1-e-d) <= bound),
                 d <= 2.0*(cp.sqrt(cp.atoms.min(e_min))-e)])

            prob.solve()

            if(e.value is None or d.value is None):
                # Only in case where the solution is not found
                return (None, None)
            else:
                e = e.value[0]
                d = d.value[0]

            c_bound = 1.0-((1-(2*e+d))**2.0)/(1-2*d)

            if(c_bound > 1.0-t):
                u = t
            else:
                l = t
        return (e, d)

    def __optimize_given_e_d(self, e, d, eS, dS, kl, m):
        """
        Optimize the posterior distribution given
        the joint error e and the disagreement d

        Parameters
        ----------
        e: float
            The (true) empirical error
        d: float
            The (true) disagreement
        eS: ndarray
            The empirical joint error
        dS: ndarray
            The empirical disagreement
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        e = torch.tensor(e, device=eS.device)
        d = torch.tensor(d, device=dS.device)

        b = self.__bound(kl, m, self.delta)
        self._loss = -self.__log_barrier(self.__kl_tri(dS, eS, d, e)-b)
        self._loss += self.__log_barrier((2.0*eS+dS)-1.0)

        # We compute the gradient descent step given (e,d)
        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

    def _optimize(self, batch):
        """
        Optimize the PAC-Bound 2 by gradient descent

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
        # and the bound
        eS = torch.mean((0.5*(1.0-y*pred))**2.0)
        dS = torch.mean(0.5*(1.0-(pred**2.0)))
        m = batch["m"]

        # We optimize the inner maximization problem
        (e, d) = self.__optimize_given_eS_dS(
            eS.item(), dS.item(), kl, m)
        # We optimize the outer minimization problem
        if(e is not None and d is not None):
            self.__optimize_given_e_d(e, d, eS, dS, kl, m)

        self._log["c-bound"] = self.__c_bound(e, d)
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
# [3] Convex Optimization
#     Stephen Boyd, Lieven Vandenberghe, 2004
