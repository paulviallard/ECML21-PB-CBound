import logging
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import warnings

from core.metrics import Metrics
from learner.c_bound_mcallester_learner import CBoundMcAllesterLearner
from learner.c_bound_seeger_learner import CBoundSeegerLearner
from learner.c_bound_joint_learner import CBoundJointLearner
from voter.stump import DecisionStumpMV
from voter.tree import TreeMV


###############################################################################


def main():
    logging.basicConfig(level=logging.INFO)
    logging.StreamHandler.terminator = ""
    warnings.filterwarnings("ignore")

    # ----------------------------------------------------------------------- #

    zero_one = Metrics("ZeroOne").fit

    def generate_MV_stump(x_train, y_train):
        majority_vote = DecisionStumpMV(
            x_train, y_train,
            nb_per_attribute=10,
            complemented=True, quasi_uniform=False)
        return x_train, y_train, majority_vote

    def generate_MV_tree(x_train, y_train):
        x_prior = x_train[:int(0.5*len(x_train)), :]
        y_prior = y_train[:int(0.5*len(y_train)), :]
        x_train = x_train[int(0.5*len(x_train)):, :]
        y_train = y_train[int(0.5*len(y_train)):, :]
        majority_vote = TreeMV(
            x_prior, y_prior,
            nb_tree=100,
            complemented=False, quasi_uniform=False)
        return x_train, y_train, majority_vote

    # ----------------------------------------------------------------------- #

    learner_dict = {
        "Alg. 1": CBoundMcAllesterLearner,
        "Alg. 2": CBoundSeegerLearner,
        "Alg. 3": CBoundJointLearner
    }
    cbound_dict = {
        "Alg. 1": "CBoundMcAllester",
        "Alg. 2": "CBoundSeeger",
        "Alg. 3": "CBoundJoint"
    }
    voter_dict = {
        "decision stumps": generate_MV_stump,
        "decision trees": generate_MV_tree,
    }
    epoch_dict = {"decision stumps": 1000, "decision trees": 100}

    # ----------------------------------------------------------------------- #

    # We generate the dataset (the "moons" dataset)
    x, y = make_moons(n_samples=1000, random_state=42)
    y[y == 0] = -1
    y = np.expand_dims(y, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    # ----------------------------------------------------------------------- #

    # For each algorithm (Alg. 1; Alg. 2 and Alg. 3)
    for algo, Learner in learner_dict.items():

        # For each voter type (i.e, decision stump or decision tree)
        for voter, generate_MV in voter_dict.items():

            logging.info("Running {} with {}\n".format(algo, voter))
            # We generate the majority vote (MV)
            x_train, y_train, majority_vote = generate_MV(x_train, y_train)

            # We learn the posterior distribution associated to the MV
            learner = Learner(
                majority_vote, epoch=epoch_dict[voter], batch_size=y.shape[0])
            learner = learner.fit(x_train, y_train)

            # We compute the train/test majority vote risk and
            # the PAC-Bayesian C-Bound
            y_p_train = learner.predict(x_train)
            y_p_test = learner.predict(x_test)
            c_bound = Metrics(
                cbound_dict[algo], majority_vote, delta=0.05).fit
            r_MV_S = zero_one(y_train, y_p_train)
            r_MV_T = zero_one(y_test, y_p_test)
            cb = c_bound(y_train, y_p_train)

            logging.info(("MV train risk: {:.4f}\n").format(r_MV_S))
            logging.info(("PAC-Bayesian C-Bound: {:.4f}\n").format(cb))
            logging.info(("MV test risk: {:.4f}\n").format(r_MV_T))


if __name__ == '__main__':
    main()
