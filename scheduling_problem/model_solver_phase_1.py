import pyomo.environ as pyo
import random
import matplotlib.pyplot as plt
import numpy as np

# Create a concrete model
model = pyo.ConcreteModel()

# REAL DATA
# Parameters (sample data generation)
MACHINES = 6  # Number of machines
MAX_JOB_N = 2  # Maximum number of jobs per machine
T_MAX = 24  # Number of time periods (e.g., 48 half-hours in a day)

# Define sets
T = list(range(1, T_MAX + 1)) # time # TODO: si parte da uno o da zero??
I = list(range(1, MACHINES + 1)) # machines
J = list(range(1, MAX_JOB_N + 1)) # jobs

model.T = pyo.Set(initialize=T)
model.I = pyo.Set(initialize=I)
model.J = pyo.Set(initialize=J)

# Cost parameters TODO
c_b = 1000  # Cost per battery
c_p = 2000   # Cost per unit of power

# Energy parameters (random generation) TODO: put real data
# e_i: energy consumption when machine i is running
e = {1 : 1000, 2: 1000, 3: 3506, 4:3502, 5: 3381, 6:8856}
# f_i: additional energy consumed when machine i starts
f = {i: 1000 for i in I}
# p_t: energy produced at time t by one unit of power
p = {t: 2000 for t in T}
# m_t: maximum energy available at time t
mmm = {t: random.randint(80000, 200000) for t in T}
# d_i: duration of job on machine i
d = {i: random.randint(3, 8) for i in I}
# n_i: number of jobs required for machine i
n_jobs = {i: 1 for i in I if i != 1} 
n_jobs[1] = 2
# c_i: cooldown period for machine i
c = {i: random.randint(2, 4) for i in I}
# THRESHOLD_FOR_JOB_J
THRESHOLD_FOR_JOB_J_AND_I = {(i,j): 12+random.randint(1,T_MAX-12) for i in I for j in J} #TODO: put j*24 as we have some jobs per day
# B: battery capacity
B = 2000

# Sets of dependencies and shared resources (example)
# Pairs of machines where the second depends on the first
M_dependencies = [(1, 2), (2, 3)]  # Machine 2 depends on 1, Machine 3 depends on 2
# Groups of machines that share resources and cannot run simultaneously
M_shared = [[1, 3], [2, 4]]  # Machines 1 and 3 share resources, as do 2 and 4
# TODO: silent times
silent_periods = {}#1: range(10, 16), 3: range(30, 36)}  # Machine 1 off during 10-15, Machine 3 off during 30-35


# Print the generated data for reference
print("Generated Parameters:")
print(f"Cost per battery (c_b): {c_b}")
print(f"Cost per power unit (c_p): {c_p}")
print(f"Machine energy usage (e): {e}")
print(f"Machine startup energy (f): {f}")
print(f"Battery capacity (B): {B}")
print(f"Job durations (d): {d}")
print(f"Required jobs per machine (n_jobs): {n_jobs}")
print(f"Machine cooldown periods (c): {c}")
print(f"Machine dependencies: {M_dependencies}")
print(f"Shared resource groups: {M_shared}")

# Variables
model.M = pyo.Var(domain=pyo.NonNegativeIntegers)  # 1 if machine i runs job j at time t
model.N = pyo.Var(domain=pyo.NonNegativeIntegers)  # 1 if machine i runs job j at time t
model.x = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # 1 if machine i runs job j at time t
model.y = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # 1 if machine i starts job j at time t
model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Energy stored at time t

# Set the jobs that a machine can't do
for i in I:
    for j in J:
        if j > n_jobs[i]:
            model.x[i, :, j].fix(0)
            model.y[i, :, j].fix(0)

# Objective: Minimize battery and power costs. m is always the model, and then I can put other things
def objective_rule(m):
    return m.N * c_b + m.M * c_p
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Constraints

# 1. Energy balance constraint
def energy_balance_rule(m, t):
    return sum(e[i] * m.x[i, t, j] + f[i] * m.y[i, t, j] for i in m.I for j in m.J) - m.M * p[t] - m.s[t] <= 0
model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

# 2. Storage computation constraint
def storage_rule(m, t):
    if t == 1: # TODO: si parte da uno o da zero??
        return m.s[t] == 0  # Assume starting with empty storage
    else:
        return m.s[t] <= sum(m.M * p[tp] - sum(e[i] * m.x[i, tp, j] + f[i] * m.y[i, tp, j] for i in m.I for j in m.J) 
                             for tp in range(1, t))
model.storage_constraint = pyo.Constraint(model.T, rule=storage_rule)

# 3. Battery capacity constraint. Non negative as it was defined before
def battery_capacity_rule(m, t):
    return m.s[t] <= m.N * B
model.battery_constraint = pyo.Constraint(model.T, rule=battery_capacity_rule)

""" 
# 4. Total usage >= needed
def usage_requirement(m, i):
    return sum(m.x[i, t, j] for t in m.T for j in m.J) >= n_jobs[i] * d[i]
model.usage_req = pyo.Constraint(model.I, rule=usage_requirement)
"""

# 5. Each machine can do one job at a time
def one_job_at_time(m, i, t):
    return sum(m.x[i, t, j] for j in m.J) <= 1
model.single_job_time = pyo.Constraint(model.I, model.T, rule=one_job_at_time)

# 6. Same for starting
def one_start_at_time(m, i, t):
    return sum(m.y[i, t, j] for j in m.J) <= 1
model.single_start_time = pyo.Constraint(model.I, model.T, rule=one_start_at_time)

# 7. Max energy constraint
def max_energy(m, t):
    return sum(e[i] * m.x[i, t, j] for i in m.I for j in m.J) <= mmm[t]
model.max_energy = pyo.Constraint(model.T, rule=max_energy)

# 8. Silent periods for machines (some machines must be off at specific times)
# For example, machine 1 must be off during time periods 10-15
model.silent_periods = pyo.ConstraintList()
for i, times in silent_periods.items():
    for t in times:
        if t <= T_MAX:  # Make sure the time is within our horizon
            model.silent_periods.add(sum(model.x[i, t, j] for j in J) == 0)

# 9. Shared resource constraint
def shared_resources(m, t, g):
    machines_in_group = [i for i in g if i in I]  # Ensure the machines are in our defined set
    if machines_in_group:
        return sum(m.x[i, t, j] for i in machines_in_group for j in m.J) <= 1
    return pyo.Constraint.Skip
model.shared_resources = pyo.ConstraintList()
for t in T:
    for group in M_shared:
        model.shared_resources.add(shared_resources(model, t, group))

# 10. Start implies run and continuation constraint
def run_constraint(m, i, t, j):
    if t == 1: #TODO: time at zero???
        # For the first time period
        return m.x[i, t, j] == m.y[i, t, j]
    else:
        # For subsequent time periods: x[i,t,j] = y[i,t,j] + x[i,t-1,j] - discontinuation
        # This simplifies to: x[i,t,j] <= y[i,t,j] + x[i,t-1,j]
        return m.x[i, t, j] <= m.y[i, t, j] + m.x[i, t-1, j] #TODO: check if this is right, but at the optimum it shouldn't do something like initializing and not using it
model.run_constraint = pyo.Constraint(model.I, model.T, model.J, rule=run_constraint)

# 10b. Start implies run and continuation constraint
def run_constraint_2(m, i, t, j):
    if t == 1: #TODO: time at zero???
        # For the first time period
        return pyo.Constraint.Skip
    else:
        # For subsequent time periods: x[i,t,j] = y[i,t,j] + x[i,t-1,j] - discontinuation
        # This simplifies to: x[i,t,j] <= y[i,t,j] + x[i,t-1,j]
        return m.y[i, t, j] <= m.x[i, t, j] 
model.run_constraint_2 = pyo.Constraint(model.I, model.T, model.J, rule=run_constraint_2)

# 11. Dependency constraint
def dependency_rule(m, k, kp1, t, j):
    if t == 1:
        return m.y[kp1, t, j] == 0  # Cannot start dependent job at first time period
    else:
        # Job on machine kp1 can start only if job on machine k was completed
        prev_completions = sum(m.x[k, tp, j] for tp in range(1, t)) #TODO: time starts at zero?
        return m.y[kp1, t, j] <= prev_completions / d[k]
model.dependencies = pyo.ConstraintList()
for (k, kp1) in M_dependencies:
    if k in I and kp1 in I:  # Ensure both machines are in our defined set
        for t in T:
            for j in J:
                model.dependencies.add(dependency_rule(model, k, kp1, t, j))

# 12. Cooldown constraint
def cooldown_rule(m, i, t):
    if t <= c[i]:
        return sum(m.y[i, t, j] for j in J) <= 1 # Allow starting in early time periods
    else:
        # Can only start if the machine was off for at least c_i time units
        cooldown_sum = sum(1 - sum(m.x[i, tp, jj] for jj in m.J) for tp in range(t-c[i], t))
        return m.y[i, t, j] <= cooldown_sum / c[i]
model.cooldowns = pyo.Constraint(model.I, model.T, rule=cooldown_rule)

# 13. Job duration enforcement
def duration_rule(m, i, j):
    return sum(m.x[i, tp, j] for tp in range(1, THRESHOLD_FOR_JOB_J_AND_I[(i,j)]+1)) == d[i]

# 13b. Job duration enforcement
def duration_rule_2(m, i, j):
    return sum(m.x[i, tp, j] for tp in range(THRESHOLD_FOR_JOB_J_AND_I[(i,j)]+1, T_MAX+1)) == 0

model.duration = pyo.ConstraintList()
for i in I:
    for j in J:
        model.duration.add(duration_rule(model, i, j))
        model.duration.add(duration_rule_2(model, i, j))

# Solve the model
solver = pyo.SolverFactory('glpk')
print("Solving the model...")
result = solver.solve(model, tee=True)  # tee=True shows the solver output

# Print results
print(f"\nSolution Status: {result.solver.status}, Termination Condition: {result.solver.termination_condition}")

if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    print(f"Objective Value: {pyo.value(model.objective)}")
    
    # Print number of panels and batteries
    print(f"\nNumber of Batteries: {pyo.value(model.N)}")
    print(f"\nNumber of Panels: {pyo.value(model.M)}")
    
    # Print machine schedules
    for i in I:
        print(f"\nMachine {i} Schedule:")
        for j in J:
            start_times = [t for t in T if pyo.value(model.y[i, t, j]) > 0.5]
            operative_times = [t for t in T if pyo.value(model.x[i, t, j]) > 0.5]
            if start_times:
                for start_t in start_times:
                    print(f"  Job {j} starts at t={start_t}")
            print("------------------------------")
            if operative_times:
                for op_t in operative_times:
                    print(f"  Job {j} operates at t={op_t}")
    
    # Print energy storage levels
    storage_values = [pyo.value(model.s[t]) for t in T]
    print("\nStorage Levels:")
    for t in T:
        print(f"  t={t}: {pyo.value(model.s[t]):.2f}")
    
    # Create visualization of the schedule
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot machine schedules
    for i in I:
        for t in T:
            for j in J:
                if pyo.value(model.x[i, t, j]) > 0.5:
                    ax1.bar(t, 1, bottom=i-1, color=f'C{j}', edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Machine')
    ax1.set_yticks(range(0, MACHINES))
    ax1.set_yticklabels([f'Machine {i}' for i in I])
    ax1.set_title('Machine Schedule')
    ax1.legend([f'Job {j}' for j in J], loc='upper right')
    
    # Plot energy storage
    ax2.plot(T, storage_values, marker='o', linestyle='-', markersize=4)
    battery_capacity = int(model.N.value) * B  # or value(model.N * B)
    ax2.axhline(y=battery_capacity, color='r', linestyle='--', label=f'Battery Capacity ({battery_capacity})')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Energy Storage')
    ax2.set_title('Energy Storage Levels')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('schedule_visualization.png')
    print("\nSchedule visualization saved as 'schedule_visualization.png'")
    
    plt.show()
else:
    print("Failed to find an optimal solution.")
