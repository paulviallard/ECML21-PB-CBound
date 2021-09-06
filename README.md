This repository contains the source code of the paper entitled:

> **Self-Bounding Majority Vote Learning Algorithms by the Direct Minimization of a Tight PAC-Bayesian C-Bound**<br/>
> Paul Viallard, Pascal Germain, Amaury Habrard, and Emilie Morvant<br/>
> ECML-PKDD 2021, 2021

This repository contains two branches: <tt>master</tt> and <tt>exp</tt>. 
The source code in the <tt>master</tt> branch is essentially the implementation of the PAC-Bayesian C-Bound minimization algorithms. Moreover, we introduce a small example (in example.py) that run our algorithms. The <tt>exp</tt> branch contains the source code to reproduce the experiments and the figures of the paper.

### Running the experiments 

##### In branch <tt>master</tt>
To run the example, you need to execute the following command in your bash shell.
```bash
python example.py
```

##### In branch <tt>exp</tt>
The datasets can be downloaded and preprocessed by running the following command in your bash shell.
```bash
./generate_data
```
Then, to run the experiments and generate the figures, you have to run the following command in your bash shell.
```bash
./run
```

### Dependencies

The code was tested on Python 3.9.1 with the packages
* cvxopt (1.2.6)
* cvxpy (1.1.14)
* h5py (3.4.0)
* matplotlib (3.4.3)
* numpy (1.21.2)
* pandas (1.3.2)
* requests (2.26.0)
* scikit_learn (0.24.2)
* seaborn (0.11.2)
* torch (1.9.0)
* torchvision (0.10.0)

These dependencies can be installed (using pip) with the following command.
> pip install cvxopt==1.2.6 cvxpy==1.1.14 h5py==3.4.0 matplotlib==3.4.3 numpy==1.21.2 pandas==1.3.2  requests==2.26.0 scikit_learn==0.24.2 seaborn==0.11.2 torch==1.9.0 torchvision==0.10.0
