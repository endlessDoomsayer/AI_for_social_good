import pyomo.environ as pyo
import numpy as np

# Create a concrete model
model = pyo.ConcreteModel()

# Define sets and parameters (these would be populated with actual data)
# T: time periods
# I: machines
# J: jobs
# M_prime: sets of machines with shared resources (list of lists)
# dependencies: list of tuples (k, k+1) representing dependencies between machines

# Sample data (replace with actual data)
T = range(1, 11)  # 10 time periods
I = range(1, 6)   # 5 machines
J = range(1, 4)   # 3 jobs
M_prime_sets = [[1, 2], [3, 4]]  # Example of machine subsets with shared resources
dependencies = [(1, 2), (3, 4)]  # Example of dependencies between machines

model.T = pyo.Set(initialize=T)
model.I = pyo.Set(initialize=I)
model.J = pyo.Set(initialize=J)

# Parameters
# e_i: resource consumption of machine i
# f_i: additional resource consumption for machine i when starting
# M: maximum resource production
# NB: maximum storage capacity
# n_i: minimum number of jobs required for machine i
# d_i: duration of job on machine i
# m_t: maximum resource consumption at time t
# c_i: cooldown time for machine i after use

# Sample parameter values (replace with actual data)
e_data = {i: 5 + i for i in I}  # Resource consumption for each machine
f_data = {i: 2 * i for i in I}  # Additional resource for starting
M_data = 25  # Maximum resource production
NB_data = 50  # Maximum storage capacity
n_data = {i: 1 for i in I}  # Minimum number of jobs per machine
d_data = {i: i % 3 + 1 for i in I}  # Duration of job on machine i
m_data = {t: 20 for t in T}  # Maximum resource consumption at time t
c_data = {i: 2 for i in I}  # Cooldown time

model.e = pyo.Param(model.I, initialize=e_data)
model.f = pyo.Param(model.I, initialize=f_data)
model.M = pyo.Param(initialize=M_data)
model.NB = pyo.Param(initialize=NB_data)
model.n = pyo.Param(model.I, initialize=n_data)
model.d = pyo.Param(model.I, initialize=d_data)
model.m = pyo.Param(model.T, initialize=m_data)
model.c = pyo.Param(model.I, initialize=c_data)

# Define which machines are unavailable at certain times
# For example, machine 2 is unavailable at time 3
unavailable = [(2, 3)]

# Define decision variables
model.x = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # Assignment variable
model.y = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # Start variable
model.z = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Objective function variable
model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, NB_data))  # Storage variable
model.p = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Production variable

# Objective function: minimize sum of z_t
model.objective = pyo.Objective(
    expr=sum(model.z[t] for t in model.T),
    sense=pyo.minimize
)

# Constraints

# Constraint 1: Resource balance constraint at every time t
@model.Constraint(model.T)
def resource_balance(model, t):
    return sum(model.e[i] * model.x[i, t, j] + model.f[i] * model.y[i, t, j] 
              for i in model.I for j in model.J) - model.M * model.p[t] - model.s[t] <= model.z[t]

# Constraint 2: Storage calculation for every time t
@model.Constraint(model.T)
def storage_calculation(model, t):
    if t == 1:
        # Initial storage is 0
        return model.s[t] == 0
    else:
        return model.s[t] == sum(model.M * model.p[t_prime] - 
                              sum(model.e[i] * model.x[i, t_prime, j] + model.f[i] * model.y[i, t_prime, j] 
                                 for i in model.I for j in model.J)
                             for t_prime in range(1, t))

# Constraint 3: Storage bounds
# Already handled by the variable bounds (0, NB_data)

# Constraint 4: z_t is non-negative
# Already handled by the variable domain

# Constraint 5: Each machine must process minimum number of jobs
@model.Constraint(model.I)
def min_jobs_per_machine(model, i):
    return sum(model.x[i, t, j] for t in model.T for j in model.J) >= model.n[i] * model.d[i]

# Constraint 6: Each machine can process at most one job at a time
@model.Constraint(model.I, model.T)
def one_job_per_machine_time(model, i, t):
    return sum(model.x[i, t, j] for j in model.J) <= 1

# Constraint 7: Each machine can start at most one job at a time
@model.Constraint(model.I, model.T)
def one_start_per_machine_time(model, i, t):
    return sum(model.y[i, t, j] for j in model.J) <= 1

# Constraint 8: Resource consumption limit at each time
@model.Constraint(model.T)
def resource_consumption_limit(model, t):
    return sum(model.e[i] * model.x[i, t, j] for i in model.I for j in model.J) <= model.m[t]

# Constraint 9: Some machines are unavailable at certain times
for i, t in unavailable:
    @model.Constraint()
    def machine_unavailable(model, i=i, t=t):
        return sum(model.x[i, t, j] for j in model.J) == 0

# Constraint 10: Machines with shared resources constraint
for M_set in M_prime_sets:
    @model.Constraint(model.T)
    def shared_resources(model, t, M_set=M_set):
        return sum(model.x[i, t, j] for i in M_set for j in model.J) <= 1

# Constraint 11: Start variable relationship with assignment variable
@model.Constraint(model.I, model.T, model.J)
def start_relationship(model, i, t, j):
    if t == 1:
        return model.y[i, t, j] == model.x[i, t, j]
    else:
        return model.y[i, t, j] <= model.x[i, t, j] - model.x[i, t-1, j] + 1

# Constraint 12: Dependencies between machines
for k, k_plus_1 in dependencies:
    @model.Constraint(model.T, model.J)
    def machine_dependencies(model, t, j, k=k, k_plus_1=k_plus_1):
        if t == 1:
            return model.y[k_plus_1, t, j] <= 0
        else:
            return model.y[k_plus_1, t, j] <= sum(model.x[k, t_prime, j] for t_prime in range(1, t)) / model.d[k]

# Constraint 13: Cooldown period for machines
@model.Constraint(model.I, model.T, model.J)
def cooldown_period(model, i, t, j):
    if t <= model.c[i]:
        return model.y[i, t, j] <= 0
    else:
        cooldown_sum = sum(1 - sum(model.x[i, t_prime, j_prime] for j_prime in model.J) 
                           for t_prime in range(t - model.c[i], t))
        return model.y[i, t, j] <= cooldown_sum / model.c[i]

# Constraint 14: Job completion constraint
@model.Constraint(model.I, model.J)
def job_completion(model, i, j):
    return sum(model.x[i, t, j] for t in model.T) == model.d[i]

# Solve the model using GLPK
solver = pyo.SolverFactory('glpk')
results = solver.solve(model, tee=True)

# Print solution status
print("Solution Status:", results.solver.status)
print("Termination Condition:", results.solver.termination_condition)

# Print objective value
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("Objective value:", pyo.value(model.objective))
    
    # Print selected variable values
    print("\nSchedule (x[i,t,j] = 1 means machine i processes job j at time t):")
    for i in model.I:
        for t in model.T:
            for j in model.J:
                if pyo.value(model.x[i, t, j]) > 0.5:  # Using 0.5 threshold for binary variables
                    print(f"Machine {i} processes job {j} at time {t}")
    
    print("\nStarts (y[i,t,j] = 1 means machine i starts processing job j at time t):")
    for i in model.I:
        for t in model.T:
            for j in model.J:
                if pyo.value(model.y[i, t, j]) > 0.5:
                    print(f"Machine {i} starts job {j} at time {t}")
    
    print("\nStorage levels (s[t]):")
    for t in model.T:
        print(f"Time {t}: {pyo.value(model.s[t])}")
    
    print("\nProduction levels (p[t]):")
    for t in model.T:
        print(f"Time {t}: {pyo.value(model.p[t])}")
    
    print("\nObjective function components (z[t]):")
    for t in model.T:
        print(f"Time {t}: {pyo.value(model.z[t])}")
