# Structure of the folder

4 models, various ways to solve them.

## Model 1
We can solve it as a MILP model or by doing binary search on both N and M. This leads to the same result, but the second
way is a lot faster

## Model 2
It's not a model, simply a division. It's done to understand the time horizon.

## Model 3
Within that time horizon, we could see if it's more helpful to set a lower number of batteries and panels and pay some energy. It can be done both with a MILP model or by finding a feasible solution and then doing local search.

## Model 4
It is the real scheduling once we fix N and M.

### 4.1
If we found out we could simply use a lot of N and M to not pay energy, let's find the optimal scheduling.

This is implemented both as a MILP model, a linear programming model and as it could be solved by a SAT solver.
We also have the enhanced version that compares some of the techniques seen in class to solve it.

### 4.3 
If we have to pay some energy. Uses MILP.