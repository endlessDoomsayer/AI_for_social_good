# Structure of the folder
This folder contains the code and the full specification of the four phases of our
system. 

To see the full specifications of the phases and the corresponding models, open the 
`main_script.ipynb` file.

Below there is a brief description of the phases and the corresponding files.

## Phase 1
Minimize the number of panels and batteries without using any external energy source.
It is solved in two ways:
- `_1_scip.py` solves it as a MIP model using SCIP;
- `_1_bin_search_scip.py` and `_1_bin_search_scip_inverted.py` solve it by doing binary search on both the number of panels (M) and the number of batteries (N). 
While the first one does a binary search on M and for each M does a binary search on N to find the minimum couple (M,N) for which the scheduling problem is feasible,
the second one first does a binary search on N and then on M aiming at the same objective.

As stated in the paper, the second method provides a good approximate solution in less than half the time of the first one.

## Phase 2
It doesn't require to solve a model, but to compute a simple division.
It's done to understand the time horizon that could be covered by only using external energy with the same cost of the result of phase 1.

`_2.py` solves it.

## Phase 3
Within the time horizon given by phase 2, we try to set a lower number of batteries and panels by buying some external energy. 
It is solved in two ways:
- `_3_scip.py` solves it as a MIP model using SCIP;
- `_3_local_search.py` first finds a feasible solution by using SCIP and then tries to optimize it locally to obtain a better result.

As stated in the paper, the second method provides a good approximate solution in far fewer time than the first one.

## Phase 4
This phase is devoted to obtain the actual scheduling of the machines once we have set the number of panels and batteries.
It consists of two subcases: 4.1 and 4.3.

### 4.1
Finds the optimal scheduling requiring to not import any external energy.
It is solved in two ways:
- `_4_1_scip.py` solves it as a MIP model using SCIP;
- `_4_1_CSP_alternatives.py` solves it with various CSP techniques using OR Tools, and then compares them. In particular,
we use a standard backtracking algorithm and an advanced one that incorporates heuristics such as Most Constrained Variable and Least Constraining Value, along with advanced methods like Back-jumping, No-Good Learning, and Constraint Propagation techniques (e.g., Arc Consistency).

In the case of modified constraints we can exploit the algorithms implemented in `_4_1_adapt.py`, which are Tabu Search and
Simulated Annealing, that take as input a scheduling and perform a local search on its neighbors to make it feasible.

### 4.3 
Finds the optimal scheduling if we can use imported energy.
`_4_3_scip.py` solves it as a MIP model using SCIP.