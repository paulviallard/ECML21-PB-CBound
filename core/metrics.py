import cvxpy as cp
import importlib
import inspect
import math
import numpy as np

from core.kl_inv import kl_inv
from voter.majority_vote import MajorityVote

###############################################################################


class MetaMetrics(type):

    def __get_class_dict(cls):
        class_dict = {}
        for class_name, class_ in inspect.getmembers(
            importlib.import_module("core.metrics"), inspect.isclass
        ):
            if(class_name != "MetaMetrics" and class_name != "Metrics"):
                class_name = class_name.replace("Metrics", "")
                class_dict[class_name] = class_
        return class_dict

    def __call__(cls, *args, **kwargs):
        # We initialize the base classes
        bases = (cls, )

        # We get the name of the module
        if("name" not in kwargs):
            class_name = args[0]
        else:
            class_name = kwargs["name"]

        # We get the module dictionary
        class_dict = cls.__get_class_dict()

        # We check that the module exists
        if(class_name not in class_dict):
            raise Exception(class_name+" doesn't exist")

        # We add the new module in the base classes
        bases = (class_dict[class_name], )+bases

        # We create the new object with the good base classes
        new_cls = type(cls.__name__, bases, {})
        return super(MetaMetrics, new_cls).__call__(*args, **kwargs)


# --------------------------------------------------------------------------- #


class Metrics(metaclass=MetaMetrics):

    def __init__(self, name, majority_vote=None):
        """
        We initialize the metric (or the objective function) to use

        Parameters
        ----------
        name: str
            The name of the metric (or the objective function)
        majority_vote: MajorityVote
            The majority vote object
        """
        super().__init__()
        self.mv = majority_vote

    def fit(self, y, y_p):
        """
        We forward in the metric or the objective function

        Parameters
        ----------
        y: ndarray
            The labels
        y_p: ndarray
            The predicted labels
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #


class RiskMetrics():

    def fit(self, y, y_p):
        assert isinstance(y, np.ndarray)
        assert isinstance(y_p, np.ndarray)

        assert (len(y_p.shape) == 2 and len(y.shape) == 2 and
                y_p.shape[0] == y.shape[0] and
                y_p.shape[1] == y.shape[1] and
                y.shape[1] == 1 and y_p.shape[0] > 0)
        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        # We compute the Gibbs Risk (see Equations (6) and (7) in [1])
        return np.mean(0.5*(1.0-y_p*y))


class DisagreementMetrics():

    def fit(self, y, y_p):
        assert isinstance(y, np.ndarray)
        assert isinstance(y_p, np.ndarray)

        assert (len(y_p.shape) == 2 and len(y.shape) == 2 and
                y_p.shape[0] == y.shape[0] and
                y_p.shape[1] == y.shape[1] and
                y.shape[1] == 1 and y_p.shape[0] > 0)
        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        # We compute the Disagreement (see Equations (8) and (9) in [1])
        return np.mean(0.5*(1.0-y_p**2.0))


class JointMetrics():

    def fit(self, y, y_p):
        assert isinstance(y, np.ndarray)
        assert isinstance(y_p, np.ndarray)

        assert (len(y_p.shape) == 2 and len(y.shape) == 2 and
                y_p.shape[0] == y.shape[0] and
                y_p.shape[1] == y.shape[1] and
                y.shape[1] == 1 and y_p.shape[0] > 0)
        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        # We compute the Joint Error (see Definition 23 in [1])
        return np.mean((0.5*(1.0-y_p*y))**2.0)


class ZeroOneMetrics():

    def fit(self, y, y_p):
        assert isinstance(y, np.ndarray)
        assert isinstance(y_p, np.ndarray)
        risk = Metrics("Risk").fit

        # We compute the risk with the 0-1 loss
        y_p_ = 2.0*(y_p > 0.0).astype(float)-1.0
        return risk(y, y_p_)


class CBoundMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(y, np.ndarray)
        assert isinstance(y_p, np.ndarray)

        risk = Metrics("Risk").fit
        disa = Metrics("Disagreement").fit

        rS = risk(y, y_p)
        dS = disa(y, y_p)

        return self.__c_bound(rS, dS)

    def __c_bound(self, r, d):
        """
        Compute the C-Bound (The "third form" in [1])

        Parameters
        ----------
        r: ndarray
            The risk
        d: ndarray
            The disagreement
        """
        r = np.minimum(0.5, r)
        d = np.maximum(0.0, d)
        cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
        return cb


class CBoundMcAllesterMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(self.mv, MajorityVote)
        assert isinstance(y, np.ndarray)
        assert isinstance(y_p, np.ndarray)

        m = y.shape[0]

        risk = Metrics("Risk").fit
        disa = Metrics("Disagreement").fit

        # We compute the empirical risk and disagreement
        rS = risk(y, y_p)
        dS = disa(y, y_p)

        prior = self.mv.prior
        post = self.mv.post

        # We compute the KL divergence
        kl = 0.0
        if(not(self.mv.quasi_uniform)):
            with np.errstate(divide='ignore'):
                kl = np.log(post/prior)
            kl[np.isinf(kl)] = 0.0
            kl[np.isnan(kl)] = 0.0
            kl = np.sum(post*kl)

        # We compute the PAC-Bayesian bounds for the risk and the disagreement
        r = self.__risk_bound(rS, kl, m)
        if(r > 0.5):
            return np.array(1.0)
        d = self.__disagreement_bound(dS, kl, m)

        # We compute the C-Bound with the PAC-Bayesian bounds
        return self.__c_bound(r, d)

    def __risk_bound(self, rS, kl, m):
        """
        Compute the (McAllester's) PAC-Bayesian bound for the risk

        Parameters
        ----------
        rS: ndarray
            The empirical risk
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        b = (1.0/(2.0*m))*(kl+math.log(
            (2.0*math.sqrt(m))/(0.5*self.delta)))
        b = rS + np.sqrt(b)
        return b

    def __disagreement_bound(self, dS, kl, m):
        """
        Compute the (McAllester's) PAC-Bayesian bound for the disagreement

        Parameters
        ----------
        dS: ndarray
            The empirical disagreement
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        b = (1.0/(2.0*m))*(2.0*kl+math.log(
            (2.0*math.sqrt(m))/(0.5*self.delta)))
        b = dS - np.sqrt(b)
        return b

    def __c_bound(self, r, d):
        """
        Compute the C-Bound (The "third form" in [1])

        Parameters
        ----------
        r: ndarray
            The risk
        d: ndarray
            The disagreement
        """
        r = np.minimum(0.5, r)
        d = np.maximum(0.0, d)
        cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
        return cb


class CBoundSeegerMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(self.mv, MajorityVote)
        assert isinstance(y, np.ndarray)
        assert isinstance(y_p, np.ndarray)

        m = y.shape[0]

        risk = Metrics("Risk").fit
        disa = Metrics("Disagreement").fit

        # We compute the empirical risk and disagreement
        rS = risk(y, y_p)
        dS = disa(y, y_p)

        # We compute the KL divergence
        prior = self.mv.prior
        post = self.mv.post
        kl = 0.0
        if(not(self.mv.quasi_uniform)):
            with np.errstate(divide='ignore'):
                kl = np.log(post/prior)
            kl[np.isinf(kl)] = 0.0
            kl[np.isnan(kl)] = 0.0
            kl = np.sum(post*kl)

        # We compute the PAC-Bayesian bounds for the risk and the disagreement
        r = self.__risk_bound(rS, kl, m)
        if(r > 0.5):
            return np.array(1.0)
        d = self.__disagreement_bound(dS, kl, m)

        # We compute the C-Bound with the PAC-Bayesian bounds
        return self.__c_bound(r, d)

    def __bound(self, kl, m):
        """
        Compute the PAC-Bayesian bound

        Parameters
        ----------
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        return (1.0/m)*(kl+math.log((2.0*math.sqrt(m))/(0.5*self.delta)))

    def __risk_bound(self, rS, kl, m):
        """
        Compute the (Seeger's) PAC-Bayesian bound for the risk

        Parameters
        ----------
        rS: ndarray
            The empirical risk
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        b = self.__bound(kl, m)
        b = kl_inv(rS, b, "MAX")
        return b

    def __disagreement_bound(self, dS, kl, m):
        """
        Compute the (Seeger's) PAC-Bayesian bound for the disagreement

        Parameters
        ----------
        dS: ndarray
            The empirical disagreement
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        b = self.__bound(2.0*kl, m)
        b = kl_inv(dS, b, "MIN")
        return b

    def __c_bound(self, r, d):
        """
        Compute the C-Bound (The "third form" in [1])

        Parameters
        ----------
        r: ndarray
            The risk
        d: ndarray
            The disagreement
        """
        r = np.minimum(0.5, r)
        d = np.maximum(0.0, d)
        cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
        return cb


class CBoundJointMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(self.mv, MajorityVote)
        assert isinstance(y, np.ndarray)
        assert isinstance(y_p, np.ndarray)

        m = y.shape[0]

        joint = Metrics("Joint").fit
        disa = Metrics("Disagreement").fit

        # We compute the empirical joint error and disagreement
        eS = joint(y, y_p)
        dS = disa(y, y_p)

        # We compute the KL divergence
        prior = self.mv.prior
        post = self.mv.post
        kl = 0.0
        if(not(self.mv.quasi_uniform)):
            with np.errstate(divide='ignore'):
                kl = np.log(post/prior)
            kl[np.isinf(kl)] = 0.0
            kl[np.isnan(kl)] = 0.0
            kl = np.sum(post*kl)

        # We compute the PAC-Bayesian bounds on
        # the joint error and the disagreement
        if(2.0*eS+dS >= 1.0 or dS > 2*(np.sqrt(eS)-eS)):
            return np.array(1.0)
        (e, d) = self.__joint_disagreement_bound(eS, dS, kl, m)

        # We compute the C-Bound with the PAC-Bayesian bounds
        return self.__c_bound(e, d)

    def __bound(self, kl, m):
        """
        Compute the PAC-Bayesian bound of PAC-Bound 2 (see page 820 of [1])

        Parameters
        ----------
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        b = math.log((2.0*math.sqrt(m)+m)/(self.delta))
        b = (1.0/m)*(2.0*kl+b)
        return b

    def __joint_disagreement_bound(self, eS, dS, kl, m, tol=0.01):
        """
        Solve the inner maximization Problem using the
        "Bisection method for quasiconvex optimization" of [2] (p 146)

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
        bound = self.__bound(kl, m)

        while(u-l > tol):
            t = (l+u)/2.0

            e = cp.Variable(shape=1, nonneg=True)
            d = cp.Variable(shape=1, nonneg=True)

            prob = cp.Problem(
                cp.Minimize((1-(2*e+d))**2.0-t*(1-2*d)),
                [(cp.kl_div(eS, e)+cp.kl_div(dS, d)
                  + cp.kl_div((1-eS-dS), 1-e-d) <= bound),
                 2*e+d <= 1,
                 d <= 2.0*(cp.sqrt(e)-e)])

            prob.solve()
            e = e.value[0]
            d = d.value[0]

            c_bound = 1.0-((1-(2*e+d))**2.0)/(1-2*d)

            if(c_bound > 1.0-t):
                u = t
            else:
                l = t
        return (e, d)

    def __c_bound(self, e, d):
        """
        Compute the C-Bound of PAC-Bound 2 (see page 820 of [1])

        Parameters
        ----------
        e: ndarray
            The joint error
        d: ndarray
            The disagreement
        """
        return (1.0-((1.0-(2.0*e+d))**2.0)/(1.0-2.0*d))


class RiskBoundMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(self.mv, MajorityVote)
        assert isinstance(y, np.ndarray)
        assert isinstance(y_p, np.ndarray)

        m = y.shape[0]

        risk = Metrics("Risk").fit

        # We compute the risk
        rS = risk(y, y_p)

        # We compute the KL divergence
        prior = self.mv.prior
        post = self.mv.post
        kl = 0.0
        if(not(self.mv.quasi_uniform)):
            with np.errstate(divide='ignore'):
                kl = np.log(post/prior)
            kl[np.isinf(kl)] = 0.0
            kl[np.isnan(kl)] = 0.0
            kl = np.sum(post*kl)

        # We compute PAC-Bayesian bound on the risk
        r = self.__risk_bound(rS, kl, m)

        # and we use the bound on the majority vote risk
        return 2.0*r

    def __bound(self, kl, m):
        """
        Compute the PAC-Bayesian bound

        Parameters
        ----------
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        return (1.0/m)*(kl+math.log((2.0*math.sqrt(m))/(self.delta)))

    def __risk_bound(self, rS, kl, m):
        """
        Compute the (Seeger's) PAC-Bayesian bound for the risk

        Parameters
        ----------
        rS: ndarray
            The empirical risk
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        b = self.__bound(kl, m)
        b = kl_inv(rS, b, "MAX")
        return b


class JointBoundMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(self.mv, MajorityVote)
        assert isinstance(y, np.ndarray)
        assert isinstance(y_p, np.ndarray)

        m = y.shape[0]

        joint = Metrics("Joint").fit

        # We compute the empirical joint error
        eS = joint(y, y_p)

        # We compute the KL divergence
        prior = self.mv.prior
        post = self.mv.post
        kl = 0.0
        if(not(self.mv.quasi_uniform)):
            with np.errstate(divide='ignore'):
                kl = np.log(post/prior)
            kl[np.isinf(kl)] = 0.0
            kl[np.isnan(kl)] = 0.0
            kl = np.sum(post*kl)

        # We compute the PAC-Bayesian on the joint error
        e = self.__risk_bound(eS, kl, m)

        # and we use the bound on the majority vote risk
        return 4.0*e

    def __bound(self, kl, m):
        """
        Compute the PAC-Bayesian bound

        Parameters
        ----------
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        return (1.0/m)*(kl+math.log((2.0*math.sqrt(m))/(self.delta)))

    def __risk_bound(self, eS, kl, m):
        """
        Compute the (Seeger's) PAC-Bayesian bound for the joint error

        Parameters
        ----------
        eS: ndarray
            The empirical risk
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        b = self.__bound(2.0*kl, m)
        b = kl_inv(eS, b, "MAX")
        return b

###############################################################################

# References:
# [1] Risk Bounds for the Majority Vote:
#     From a PAC-Bayesian Analysis to a Learning Algorithm
#     Pascal Germain, Alexandre Lacasse, François Laviolette,
#     Mario Marchand, Jean-Francis Roy, 2015
# [2] Convex Optimization
#     Stephen P. Boyd, Lieven Vandenberghe, 2004
