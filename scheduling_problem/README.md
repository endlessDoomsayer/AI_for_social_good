# Structure of the folder
This folder contains the code and the full specification of the four phases of our
system.

To see the full specifications of the phases and the corresponding models, open the 
`main_script.ipynb` file.

Below there is first an explanation of the complete system, and then a brief description of the phases and the corresponding files.

### Complete system
The complete system can be run in two ways:
- `main.py`: this script executes the four processing phases sequentially, saving the results to the appropriate files. It also leverages `run_models.py` during execution.
The number of days to set the time horizon for the model can be changed (right now is set to 1). Increasing the number of days increases the complexity and the
time required to find a solution exponentially. Some output examples are inside `output_1day/` and `output_2days/`
- `main_day_per_day.py`: this script executes phase from 1 to 3 singularly for a sequence of days, printing all the results.
It exploits a time decomposition of the problem, solving the models for smaller periods to avoid the time complexity, with the implicit assumption that job deadlines are set to the end 
of each day. A good approximation of the number of batteries and panels that are needed in phase 1 and 3 can be obtained by the worst case of the results of the days. An example of output can be found in `output_day_per_day/`

### Utilities
The project includes several utility components:
- `combine_data.py`: this script merges data from NILM and the solar panel predictor to generate comprehensive datasets. It can produce data for both industrial and residential scenarios, including modified versions that can be useful for phase 4.1 as described below.
- `output_1day/` folder: contains sample results generated using example data for one single day and `main.py`.
- `output_2days/` folder: contains sample results generated using example data for two days and `main.py`.
- `output_day_per_day/` folder: contains sample results generated using example data for one day and `main_day_per_day.py`.

## Phase 1
Minimize the number of panels and batteries without using any external energy source.
It is solved in two ways:
- `_1_scip.py` solves it as a MIP model using SCIP;
- `_1_bin_search_scip.py` and `_1_bin_search_scip_inverted.py` solve it by doing binary search on both the number of panels (M) and the number of batteries (N). 
While the first one does a binary search on M and for each M does a binary search on N to find the minimum couple (M,N) for which the scheduling problem is feasible,
the second one first does a binary search on N and then on M aiming at the same objective.

As stated in the paper, the second method provides a good approximate solution, but often requires more time than the first one.

## Phase 2
It doesn't require to solve a model, but to compute a simple division.
It's done to understand the time horizon that could be covered by only using external energy with the same cost of the result of phase 1.

`_2.py` solves it.

## Phase 3
Within the time horizon given by phase 2, we try to set a lower number of batteries and panels by also buying some external energy. 
It is solved in two ways:
- `_3_scip.py` solves it as a MIP model using SCIP;
- `_3_local_search.py` first finds a feasible solution by using SCIP and then tries to optimize it locally to obtain a better result.

As stated in the paper, the second method provides a good approximate solution in less time than the first one, which is really important especially when the number of
days increases (that's why that is the only method used in `output_2days/`).

## Phase 4
This phase is devoted to obtain the actual scheduling of the machines once we have set the number of panels and batteries.
It consists of two subcases: 4.1 and 4.3.

### 4.1
Finds the optimal scheduling requiring to not import any external energy.
It is solved in two ways:
- `_4_1_scip.py` solves it as a MIP model using SCIP;
- `_4_1_CSP_alternatives.py` solves it with various CSP techniques using OR Tools, and then compares them. In particular,
we use a standard backtracking algorithm and an advanced one that incorporates heuristics such as Most Constrained Variable and Least Constraining Value, along with advanced methods like Back-jumping, No-Good Learning, and Constraint Propagation techniques (e.g., Arc Consistency).

In the case of modified constraints (like the one given by the `get_modified_data` function inside of `combine_data.py`) we can exploit the algorithms implemented in `_4_1_adapt.py`, which are Tabu Search and
Simulated Annealing, that take as input an infeasible scheduling and perform a local search on its neighbors to make it feasible.
By doing so, we found out that Tabu Search really often finds one almost perfect solution faster.

### 4.3 
Finds the optimal scheduling if we can use imported energy.
`_4_3_scip.py` solves it as a MIP model using SCIP.