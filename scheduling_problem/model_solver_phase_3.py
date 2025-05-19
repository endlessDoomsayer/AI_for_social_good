import pyomo.environ as pyo
import pandas as pd
import numpy as np

# Create the model
model = pyo.ConcreteModel()

# Parameters (these would typically be loaded from data)
# Sets
T = list(range(1, 11))  # Time periods
I = list(range(1, 6))   # Machines
J = list(range(1, 4))   # Jobs

# Cost parameters
c_b = 100  # Cost per unit of battery capacity
c_p = 50   # Cost per unit of power capacity
N = 1      # Number of batteries
M = 1      # Power capacity factor

# Machine parameters
e = {i: 10 + i*2 for i in I}  # Energy consumption rate for machine i
f = {i: 5 + i for i in I}     # Fixed energy cost for machine i
n = {i: 1 for i in I}         # Minimum number of jobs required for machine i
d = {i: 2 for i in I}         # Duration required for machine i to complete a job
m = {t: 50 for t in T}        # Maximum energy production at time t

# Define sets for machine dependencies and shared resources
machine_dependencies = [(1, 2), (2, 3)]  # Machine k must finish before machine k+1 can start
shared_resources = [[1, 2], [3, 4]]      # Groups of machines that share resources

# Maximum battery capacity
B = 100

# Variables
model.N = pyo.Var(within=pyo.NonNegativeReals)  # Battery capacity
model.M = pyo.Var(within=pyo.NonNegativeReals)  # Power capacity
model.z = pyo.Var(T, within=pyo.NonNegativeReals)  # Power deficit at time t
model.p = pyo.Var(T, within=pyo.NonNegativeReals)  # Power production at time t
model.s = pyo.Var(T, within=pyo.NonNegativeReals)  # Energy storage at time t
model.x = pyo.Var(I, T, J, within=pyo.Binary)      # Machine i operates for job j at time t
model.y = pyo.Var(I, T, J, within=pyo.Binary)      # Machine i starts job j at time t

# Objective function: minimize total cost
model.objective = pyo.Objective(
    expr=N * c_b * model.N + M * c_p * model.M + sum(model.z[t] for t in T),
    sense=pyo.minimize
)

# Constraints

# Power deficit constraint at every time t
def power_deficit_rule(model, t):
    return sum(e[i] * model.x[i, t, j] + f[i] * model.y[i, t, j] for i in I for j in J) - M * model.p[t] - model.s[t] <= model.z[t]
model.power_deficit = pyo.Constraint(T, rule=power_deficit_rule)

# Energy storage update at each time t
def storage_update_rule(model, t):
    if t == T[0]:
        # Initial condition
        return model.s[t] == 0
    else:
        # Update based on previous periods
        return model.s[t] == sum(M * model.p[t_prev] - sum(e[i] * model.x[i, t_prev, j] + f[i] * model.y[i, t_prev, j] 
                                                       for i in I for j in J) 
                              for t_prev in range(1, t))
model.storage_update = pyo.Constraint(T, rule=storage_update_rule)

# Storage capacity limits
def storage_limit_rule(model, t):
    return model.s[t] <= N * B
model.storage_limit = pyo.Constraint(T, rule=storage_limit_rule)

# Minimum job requirements for each machine
def min_jobs_rule(model, i):
    return sum(model.x[i, t, j] for t in T for j in J) >= n[i] * d[i]
model.min_jobs = pyo.Constraint(I, rule=min_jobs_rule)

# Machine can work on at most one job at a time
def one_job_at_time_rule(model, i, t):
    return sum(model.x[i, t, j] for j in J) <= 1
model.one_job_at_time = pyo.Constraint(I, T, rule=one_job_at_time_rule)

# A job can start at most once at each time for each machine
def one_start_at_time_rule(model, i, t):
    return sum(model.y[i, t, j] for j in J) <= 1
model.one_start_at_time = pyo.Constraint(I, T, rule=one_start_at_time_rule)

# Energy production limit at each time
def energy_production_limit_rule(model, t):
    return sum(e[i] * model.x[i, t, j] for i in I for j in J) <= m[t]
model.energy_production_limit = pyo.Constraint(T, rule=energy_production_limit_rule)

# Some machines cannot operate at specific times (for example, machine 1 at time 3)
def machine_unavailable_rule(model, i=1, t=3):
    if i is None:
        i = 1
    return sum(model.x[i, t, j] for j in J) == 0
model.machine_unavailable = pyo.Constraint(rule=machine_unavailable_rule)

# Shared resource constraints
for sr in shared_resources:
    for t in T:
        model.add_component(f'shared_resource_{sr}_{t}', 
                           pyo.Constraint(expr=sum(model.x[i, t, j] for i in sr for j in J) <= 1))

# Job continuity constraint: a job either continues from previous time or starts
def job_continuity_rule_1(model, i, t, j):
    if t == T[0]:
        return model.y[i, t, j] == model.x[i, t, j]
    else:
        return model.y[i, t, j] <= model.x[i, t, j]
def job_continuity_rule_2(model, i, t, j):
    if t == T[0]:
        return model.y[i, t, j] == model.x[i, t, j]
    else:
        return model.x[i, t, j] <= model.y[i, t, j] + model.x[i, t-1, j]
model.job_continuity_1 = pyo.Constraint(I, T, J, rule=job_continuity_rule_1)
model.job_continuity_2 = pyo.Constraint(I, T, J, rule=job_continuity_rule_2)

# Machine dependency constraints
for k, k_plus_1 in machine_dependencies:
    for j in J:
        for t in T:
            model.add_component(f'dependency_{k}_{k_plus_1}_{j}_{t}',
                               pyo.Constraint(expr=model.y[k_plus_1, t, j] <= 
                                            sum(model.x[k, t_prev, j] for t_prev in range(1, t)) / d[k]))

# Cooldown constraint: machine must rest for c_i time units after completing a job
c_i = {i: 1 for i in I}  # Cooldown periods for each machine
def cooldown_rule(model, i, t, j):
    if t <= c_i[i]:
        return pyo.Constraint.Feasible
    else:
        return model.y[i, t, j] <= sum(1 - sum(model.x[i, t_prev, j_prime] for j_prime in J) 
                                     for t_prev in range(t-c_i[i], t)) / c_i[i]
model.cooldown = pyo.Constraint(I, T, J, rule=cooldown_rule)

# Job completion constraint: each job must be completed within specified duration
def job_completion_rule(model, i, j):
    return sum(model.x[i, t, j] for t in T) == d[i]
model.job_completion = pyo.Constraint(I, J, rule=job_completion_rule)

# Solve the model with GLPK
solver = pyo.SolverFactory('glpk')
results = solver.solve(model, tee=True)

# Check solution status
print("Solver Status:", results.solver.status)
print("Termination Condition:", results.solver.termination_condition)

# Display results if optimal solution found
if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("Optimal Solution Found!")
    print("Objective Value:", pyo.value(model.objective))
    print("\nBattery Capacity (N):", pyo.value(model.N))
    print("Power Capacity (M):", pyo.value(model.M))
    
    print("\nPower Production (p_t):")
    for t in T:
        print(f"Time {t}: {pyo.value(model.p[t])}")
    
    print("\nEnergy Storage (s_t):")
    for t in T:
        print(f"Time {t}: {pyo.value(model.s[t])}")
    
    print("\nPower Deficit (z_t):")
    for t in T:
        print(f"Time {t}: {pyo.value(model.z[t])}")
    
    print("\nMachine Job Assignments (x_itj = 1):")
    for i in I:
        for t in T:
            for j in J:
                if pyo.value(model.x[i, t, j]) > 0.5:
                    print(f"Machine {i}, Time {t}, Job {j}")
    
    print("\nJob Start Times (y_itj = 1):")
    for i in I:
        for t in T:
            for j in J:
                if pyo.value(model.y[i, t, j]) > 0.5:
                    print(f"Machine {i}, Time {t}, Job {j}")
else:
    print("No optimal solution found. Check model constraints and parameters.")
