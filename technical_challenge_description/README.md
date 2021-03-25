# Welcome to the Hi!ckathon !

If you've made it here, this means you've had no trouble logging in to the Hi!Paris Data Factory, and congrats for that ! It's time to get started for the real challenge.

* [Context and mission](#context-and-mission)
* [3 Main paths to win the Challenge](#3-main-paths-to-win-the-challenge)
* [Your submission and its evaluation](#your-submission-and-its-evaluation)
* [Working with Pymgrid](#working-with-pymgrid)
* [Retrieving the project](#retrieving-the-project)
* [Local work setp](#local-work-setup)

## Context and mission

### The Context 

You have 3 buildings which act as Smartgrids. Each building is equipped with a PV (solar power), a battery to store power, and the ability to import electricity from public grid. The third building is additionally equipped with a fuel-based generator (or genset), but is more tricky (you will have more details in the code).

The buildings must try to stay as independent as possible energy-wise from the public grid, but they may need at some moments to import electricity from it, which generates additional costs. Also, if the genset is used, additional costs are incurred because using fuel is expensive. 

The 3 buildings need to be supplied with energy **every hour**, and so you will have to make a decision every hour to supply them. Here are some examples of decisions you can take : 
* Should I import all the energy needed from the grid ? 
* Should I use the energy I have already stored in my battery to avoid importation costs ? 
* Is it the right time to use the fuel-based generator ? 

### Your Mission

Your task will be to design an algorithm that makes such decisions **every hour during a one-year period** while making sure that : 

* The 3 buildings are as economically profitable as possible, this implies that your algorithm can balance short term needs of energy while predicting when the demand will be at its peak to not suffer high costs due to unpreparation.
* Building and deploying your algorithm is as frugal/green/planet-friendly as possible.

## 3 Main paths to win the Challenge

The Microgrid Energy Management Use Case can be addressed with a variety of approaches, we've prepared content for 3 approaches available to you and they can all lead to victory ! : 

- **The Reinforcement Learning Path**: Compete on a Discrete environment to have the best "Simple" RL approach. 
- **The Deep Reinforcement Learning Path**: Compete on a Continuous environment and use any Deep Learning based RL approach you'd like  
- **The Expertise Path**: Well crafted Rule Based algorithms can sometimes defeat RL algorithms ! Gather business knowledge about Smart grids to implement simple but robust rules for energy management 

**Note 1:** You are not forced to strictly follow these approaches, although they are the most accessible ones given the structure of the code. If you have an idea you'd like to explore (for example, combining traditional Machine Learning approaches with any of the methods above) then reach out to the coaches to see how we can help !

**Note 2:** It is also possible to adress the problem as an Optimization under constrains problem. However, since there's a huge variety of ways you can formulate an optimization problem, make sure you know what you are doing since the coachs won't be able to guarantee that you're moving in the right direction or not. 

## Your submission and its evaluation

### How is Economic Profitability evaluated

Once your algorithm has been designed (and trained in case of RL based approaches) you will evaluate its profitability on a "test" version of the 3 buildings. You will run your algorithm on those 3 test buildings, each decision will generate a cost. **Higher profitability means a lower total cost for each building.**

The final Profitability metric will be **the sum of the the total costs for each "test" building.**

For each of the 3 buildings, you will find the scores given by a "crystal ball" algorithm. This will allow you to assess how good your strategy is as you design and improve it:

Setting | Building 1 | Building 2 | Building 3 
--- | --- | --- | --- 
Perfect Train Profitability (in €)  | 4 068.5 | 13 568.92 | 15 345.97
Perfect Test Profitability  (in €) | 3 667.98 | 12 227.57 | 13 693.58

### How is Frugality evaluated

Frugality will be evaluated using two quantities :
* **Training CPU time**: This more or less describes how much CPU power (and thus energy) was needed to train your algorithm
* **Inference CPU time**: Measures if it is heavy (thus not green) for your algorithm to be deployed and to make real-time decisions 

Adding these two numbers will give us the Total CPU Run Time, **minimizing this metric means increasing your Frugality.**

### What should my solution look like ?

**We've provided you a Template Notebook guiding you through the steps of what your submission should contain to be correctly evaluated. We strongly recommend that you follow it.**


## Working with Pymgrid

### How will I interact with the buildings in Python ?

We will be working with a Smartgrid Simulator called Pymgrid !

Pymgrid is an open-source python package developped by Total R&D to simulate Smartgrids (or Microgrids) and track their profitability over a year. The package is specifically designed to use of Reinforcement Learning, or Rule Based methods or Optimization approaches and compare them in terms of profitability and performance.

While the Pymgrid package can generate many kinds of microgrids, **the 3 buildings you will be working with are specific to this Hi!ckathon, you won't be able to find them on the public GitHub.**

Pymgrid can be found on GitHub through this link : [https://github.com/Total-RD/pymgrid](https://github.com/Total-RD/pymgrid)

**Before you install Pymgrid on your computer, complete reading this README first**

### How do I get started with Pymgrid for this challenge ?

**We've provided you with a Getting Started notebook in this repo, it will take you through the first steps to get used to Pymgrid depending on the type of approach (Rule Based, RL, Deep RL, Optimization) you choose to take.**

If you don't understand some features of Pymgrid or how to use it after reading the Getting Started notebook, don't hesitate to check out the tutorial notebooks in the Pymgrid Github's notebooks folder.

## Local work setup


### To work on a local computer: MacOS users

Before you install Pymgrid : 

0. Install git (if not already done)  : [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
1. Clone the repository : `git clone https://gitlab.repositories.hiparis.hfactory.io/team-XX/team-XX-documentation.git` replacing XX by your team number
2. Install Poetry : `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`
3. Move to the repo : `cd hiparis-hackathon-test`
4. `brew install openblas`
5. `brew install lapack`
4. Install the libraries : `OPENBLAS="$(brew --prefix openblas)" poetry install`(MacOS)
5. Launch Jupyter `poetry run jupyter notebook`

Then install Pymgrid with the following 

```bash
pip install git+https://github.com/Total-RD/pymgrid/
```

### To work on a local computer : Linux users (Tested on Ubuntu)

You can directly install pymgrid with the following line of code without any trouble : 

```bash
pip install git+https://github.com/Total-RD/pymgrid/
```

### To work on a local computer: Windows users

**We strongly recommend that you work directly on your DataFactory JupyterHub instance !** As installing the necessary tools on Windows can be very painstaking.