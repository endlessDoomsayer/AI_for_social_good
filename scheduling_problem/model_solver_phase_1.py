import pyomo.environ as pyo
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a concrete model
model = pyo.ConcreteModel()

# Parameters (sample data generation)
N = 5  # Number of batteries
M = 3  # Number of machines
MAX_JOB_N = 4  # Maximum number of jobs per machine
T_MAX = 48  # Number of time periods (e.g., 48 half-hours in a day)

# Define sets
T = list(range(1, T_MAX + 1))
I = list(range(1, M + 1))
J = list(range(1, MAX_JOB_N + 1))

model.T = pyo.Set(initialize=T)
model.I = pyo.Set(initialize=I)
model.J = pyo.Set(initialize=J)

# Cost parameters
c_b = 100  # Cost per battery
c_p = 20   # Cost per unit of power

# Energy parameters (random generation)
# e_i: energy consumption when machine i is running
e = {i: random.randint(2, 5) for i in I}
# f_i: additional energy consumed when machine i starts
f = {i: random.randint(1, 3) for i in I}
# p_t: energy price at time t
p = {t: random.uniform(0.8, 1.2) * (1 + 0.5 * np.sin(2 * np.pi * t / 24)) for t in T}  # Daily price cycle
# m_t: maximum energy available at time t
mmm = {t: random.randint(15, 20) for t in T}
# d_i: duration of job on machine i
d = {i: random.randint(3, 6) for i in I}
# n_i: number of jobs required for machine i
n_jobs = {i: random.randint(1, MAX_JOB_N) for i in I}
# c_i: cooldown period for machine i
c = {i: random.randint(2, 4) for i in I}
# B: battery capacity
B = 20

# Sets of dependencies and shared resources (example)
# Pairs of machines where the second depends on the first
M_dependencies = [(1, 2), (2, 3)]  # Machine 2 depends on 1, Machine 3 depends on 2
# Groups of machines that share resources and cannot run simultaneously
M_shared = [[1, 3], [2, 4]]  # Machines 1 and 3 share resources, as do 2 and 4

# Print the generated data for reference
print("Generated Parameters:")
print(f"Batteries (N): {N}, Cost per battery (c_b): {c_b}")
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
model.x = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # 1 if machine i runs job j at time t
model.y = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # 1 if machine i starts job j at time t
model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Energy stored at time t

# Objective: Minimize battery and power costs
def objective_rule(m):
    return N * c_b + M * c_p
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Constraints

# 1. Energy balance constraint
def energy_balance_rule(m, t):
    return sum(e[i] * m.x[i, t, j] + f[i] * m.y[i, t, j] for i in m.I for j in m.J) - M * p[t] - m.s[t] <= 0
model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

# 2. Storage computation constraint
def storage_rule(m, t):
    if t == 1:
        return m.s[t] == 0  # Assume starting with empty storage
    else:
        return m.s[t] == sum(M * p[tp] - sum(e[i] * m.x[i, tp, j] + f[i] * m.y[i, tp, j] for i in m.I for j in m.J) 
                             for tp in range(1, t))
model.storage_constraint = pyo.Constraint(model.T, rule=storage_rule)

# 3. Battery capacity constraint
def battery_capacity_rule(m, t):
    return m.s[t] <= N * B
model.battery_constraint = pyo.Constraint(model.T, rule=battery_capacity_rule)

# 4. Total usage >= needed
def usage_requirement(m, i):
    return sum(m.x[i, t, j] for t in m.T for j in m.J) >= n_jobs[i] * d[i]
model.usage_req = pyo.Constraint(model.I, rule=usage_requirement)

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
silent_periods = {1: range(10, 16), 3: range(30, 36)}  # Machine 1 off during 10-15, Machine 3 off during 30-35
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
    if t == 1:
        # For the first time period
        return m.x[i, t, j] == m.y[i, t, j]
    else:
        # For subsequent time periods: x[i,t,j] = y[i,t,j] + x[i,t-1,j] - discontinuation
        # This simplifies to: x[i,t,j] <= y[i,t,j] + x[i,t-1,j]
        return m.x[i, t, j] <= m.y[i, t, j] + m.x[i, t-1, j]
model.run_constraint = pyo.Constraint(model.I, model.T, model.J, rule=run_constraint)

# 11. Dependency constraint
def dependency_rule(m, k, kp1, t, j):
    if t == 1:
        return m.y[kp1, t, j] == 0  # Cannot start dependent job at first time period
    else:
        # Job on machine kp1 can start only if job on machine k was completed
        prev_completions = sum(m.x[k, tp, j] for tp in range(1, t))
        return m.y[kp1, t, j] <= prev_completions / d[k]
model.dependencies = pyo.ConstraintList()
for (k, kp1) in M_dependencies:
    if k in I and kp1 in I:  # Ensure both machines are in our defined set
        for t in T:
            for j in J:
                model.dependencies.add(dependency_rule(model, k, kp1, t, j))

# 12. Cooldown constraint
def cooldown_rule(m, i, t, j):
    if t <= c[i]:
        return m.y[i, t, j] <= 1  # Allow starting in early time periods
    else:
        # Can only start if the machine was off for at least c_i time units
        cooldown_sum = sum(1 - sum(m.x[i, tp, jj] for jj in m.J) for tp in range(t-c[i], t))
        return m.y[i, t, j] <= cooldown_sum / c[i]
model.cooldowns = pyo.Constraint(model.I, model.T, model.J, rule=cooldown_rule)

# 13. Job duration enforcement
def duration_rule(m, i, j, t):
    if t <= T_MAX - d[i] + 1:  # Only apply for jobs that can complete within the horizon
        # If the job starts at time t, it must run for exactly d[i] time units
        return sum(m.x[i, tp, j] for tp in range(t, t + d[i])) >= d[i] * m.y[i, t, j]
    return pyo.Constraint.Skip
model.duration = pyo.ConstraintList()
for i in I:
    for j in J:
        for t in T:
            model.duration.add(duration_rule(model, i, j, t))

# Solve the model
solver = pyo.SolverFactory('glpk')
print("Solving the model...")
result = solver.solve(model, tee=True)  # tee=True shows the solver output

# Print results
print(f"\nSolution Status: {result.solver.status}, Termination Condition: {result.solver.termination_condition}")

if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    print(f"Objective Value: {pyo.value(model.objective)}")
    
    # Print machine schedules
    for i in I:
        print(f"\nMachine {i} Schedule:")
        for j in J:
            start_times = [t for t in T if pyo.value(model.y[i, t, j]) > 0.5]
            if start_times:
                for start_t in start_times:
                    print(f"  Job {j} starts at t={start_t}, runs for {d[i]} time units")
    
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
    ax1.set_yticks(range(0, M))
    ax1.set_yticklabels([f'Machine {i}' for i in I])
    ax1.set_title('Machine Schedule')
    ax1.legend([f'Job {j}' for j in J], loc='upper right')
    
    # Plot energy storage
    ax2.plot(T, storage_values, marker='o', linestyle='-', markersize=4)
    ax2.axhline(y=N*B, color='r', linestyle='--', label=f'Battery Capacity ({N*B})')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Energy Storage')
    ax2.set_title('Energy Storage Levels')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('schedule_visualization.png')
    print("\nSchedule visualization saved as 'schedule_visualization.png'")
else:
    print("Failed to find an optimal solution.")
