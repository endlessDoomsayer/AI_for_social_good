import pyomo.environ as pyo
import matplotlib.pyplot as plt
from combine_data import get_data
import time

# Output of phase 1
M = 785
N = 50

# Create a concrete model
model = pyo.ConcreteModel()

# Get data
data = get_data()

I = data["I"]
J = data["J"]
T = data["T"]
n_jobs = data["n_jobs"]
d = data["d"]
e = data["e"]
f = data["f"]
c_b = data["c_b"]
c_p = data["c_p"]
c = data["c"]
p = data["p"]
mmm = data["mmm"]
silent_periods = data["silent_periods"]
M_shared = data["M_shared"]
M_dependencies = data["M_dependencies"]
B = data["B"]
T_MAX = data["T_MAX"]
THRESHOLD_FOR_JOB_J_AND_I = data["THRESHOLD_FOR_JOB_J_AND_I"]
MACHINES = data["MACHINES"]

# Sets
model.T = pyo.Set(initialize=T)
model.I = pyo.Set(initialize=I)
model.J = pyo.Set(initialize=J)

# Variables
model.x = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # 1 if machine i runs job j at time t
model.y = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # 1 if machine i starts job j at time t
model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Energy stored at time t

# Fix variables for jobs that a machine can't do
for i in I:
    for j in J:
        if j > n_jobs[i]:
            for t in T:
                model.x[i, t, j].fix(0)
                model.y[i, t, j].fix(0)


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
        return m.s[t] <= sum(M * p[tp] - sum(e[i] * m.x[i, tp, j] + f[i] * m.y[i, tp, j] for i in m.I for j in m.J)
                             for tp in range(1, t))


model.storage_constraint = pyo.Constraint(model.T, rule=storage_rule)


# 3. Battery capacity constraint
def battery_capacity_rule(m, t):
    return m.s[t] <= N * B


model.battery_constraint = pyo.Constraint(model.T, rule=battery_capacity_rule)

# 4. Each job must run for required duration - FIXED VERSION
# Each required job should run for exactly d[i] time periods
model.job_requirements = pyo.ConstraintList()
for i in I:
    for j in J:
        if j <= n_jobs[i]:  # Only add constraint for required jobs
            model.job_requirements.add(
                sum(model.x[i, t, j] for t in T) == d[i]
            )


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
model.silent_periods = pyo.ConstraintList()
for i, times in silent_periods.items():
    for t in times:
        if t <= T_MAX:  # Make sure the time is within our horizon
            model.silent_periods.add(sum(model.x[i, t, j] for j in J) == 0)

# 9. Shared resource constraint
model.shared_resources = pyo.ConstraintList()
for t in T:
    for group in M_shared:
        machines_in_group = [i for i in group if i in I]  # Ensure the machines are in our defined set
        if machines_in_group:
            model.shared_resources.add(
                sum(model.x[i, t, j] for i in machines_in_group for j in J) <= 1
            )


# 10. Start implies run and ensure continuity constraint - FIXED VERSION
def run_start_relation(m, i, t, j):
    if t == 1:
        # For the first time period: x[i,t,j] = y[i,t,j]
        return m.x[i, t, j] == m.y[i, t, j]
    else:
        # For subsequent periods: x[i,t,j] = y[i,t,j] + x[i,t-1,j] - end[i,t,j]
        # Since we don't model end explicitly, we allow discontinuity
        return m.x[i, t, j] <= m.y[i, t, j] + m.x[i, t - 1, j]


model.run_start_relation = pyo.Constraint(model.I, model.T, model.J, rule=run_start_relation)


# 10b. When you start a job, you must run it
def start_implies_run(m, i, t, j):
    return m.y[i, t, j] <= m.x[i, t, j]


model.start_implies_run = pyo.Constraint(model.I, model.T, model.J, rule=start_implies_run)

# 11. Dependency constraint
model.dependencies = pyo.ConstraintList()
for (k, kp1) in M_dependencies:
    if k in I and kp1 in I:  # Ensure both machines are in our defined set
        for t in T:
            for j in J:
                if t == 1:
                    model.dependencies.add(model.y[kp1, t, j] == 0)  # Cannot start dependent job at first time period
                else:
                    # Job on machine kp1 can start only if job on machine k was completed
                    prev_completions = sum(model.x[k, tp, j] for tp in range(1, t))
                    model.dependencies.add(model.y[kp1, t, j] <= prev_completions / d[k])

# 12. Cooldown constraint - FIXED VERSION
model.cooldowns = pyo.ConstraintList()
for i in I:
    for t in T:
        for j in J:
            if t <= c[i]:
                # Allow starting in early time periods
                continue  # No constraint needed
            else:
                # Can only start if the machine was off for at least c_i time units
                cooldown_sum = sum(1 - sum(model.x[i, tp, jj] for jj in J) for tp in range(t - c[i], t))
                model.cooldowns.add(model.y[i, t, j] <= cooldown_sum / c[i])

# 13. Job must be completed before threshold - FIXED VERSION
model.job_completion = pyo.ConstraintList()
for i in I:
    for j in J:
        if j <= n_jobs[i]:  # Only apply for required jobs
            # Ensure job is not scheduled after threshold
            for t in range(THRESHOLD_FOR_JOB_J_AND_I[(i, j)] + 1, T_MAX + 1):
                model.x[i, t, j].fix(0)

"""

# 14. Job continuity constraint - FIXED VERSION
# A job, once started, must run for consecutive time periods
model.job_continuity = pyo.ConstraintList()
for i in I:
    for j in J:
        if j <= n_jobs[i]:  # Only apply for required jobs
            # Each job must have exactly one start
            model.job_continuity.add(
                sum(model.y[i, t, j] for t in T) == 1
            )

            # Ensure continuous operation for the duration
            for t in range(2, T_MAX+1):
                # If the job is running at time t but not at t-1, then it must have started at t
                model.job_continuity.add(
                    model.x[i, t, j] - model.x[i, t-1, j] <= model.y[i, t, j]
                )
"""

# Solve the model
solver = pyo.SolverFactory('glpk')
print("Solving the model...")

start = time.time()
result = solver.solve(model, tee=True)  # tee=True shows the solver output
end = time.time()

print(f"Solved in {end - start:.2f} seconds")

# Print results
print(f"\nSolution Status: {result.solver.status}, Termination Condition: {result.solver.termination_condition}")

if result.solver.termination_condition == pyo.TerminationCondition.optimal:

    # Print machine schedules
    for i in I:
        print(f"\nMachine {i} Schedule:")
        for j in J:
            start_times = [t for t in T if pyo.value(model.y[i, t, j]) > 0.5]
            operative_times = [t for t in T if pyo.value(model.x[i, t, j]) > 0.5]
            if start_times:
                print(f"  Job {j} starts at t={start_times}")
            if operative_times:
                print(f"  Job {j} operates at t={operative_times}")

    # Print energy storage levels
    storage_values = [pyo.value(model.s[t]) for t in T]
    print("\nStorage Levels:")
    for t in range(1, min(11, T_MAX + 1)):  # Show first 10 time periods for brevity
        print(f"  t={t}: {pyo.value(model.s[t]):.2f}")

    # Create visualization of the schedule
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot machine schedules
    for i in I:
        for t in T:
            for j in J:
                if pyo.value(model.x[i, t, j]) > 0.5:
                    ax1.bar(t, 1, bottom=i - 1, color=f'C{j}', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Machine')
    ax1.set_yticks(range(0, MACHINES))
    ax1.set_yticklabels([f'Machine {i}' for i in I])
    ax1.set_title('Machine Schedule')
    ax1.legend([f'Job {j}' for j in J], loc='upper right')

    # Plot energy storage
    ax2.plot(T, storage_values, marker='o', linestyle='-', markersize=4)
    battery_capacity = N * B
    ax2.axhline(y=battery_capacity, color='r', linestyle='--', label=f'Battery Capacity ({battery_capacity})')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Energy Storage')
    ax2.set_title('Energy Storage Levels')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('schedule_visualization.svg', format="svg")
    print("\nSchedule visualization saved as 'schedule_visualization.svg'")

    plt.show()
else:
    print("Failed to find an optimal solution.")