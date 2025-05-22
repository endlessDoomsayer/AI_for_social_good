import pyomo.environ as pyo
import random
import matplotlib.pyplot as plt

from combine_data import get_data

# Create a concrete model
model = pyo.ConcreteModel()

BIG_M = 1000000

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
model.M = pyo.Var(domain=pyo.NonNegativeIntegers)  # Number of power units
model.N = pyo.Var(domain=pyo.NonNegativeIntegers)  # Number of batteries
model.x = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # 1 if machine i runs job j at time t
model.y = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # 1 if machine i starts job j at time t
model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Energy stored at time t

# New variables wrt model 1
model.z = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Deficit variable
model.V = pyo.Var(model.T, domain=pyo.Reals)  # Volume variable before applying constraints
model.b = pyo.Var(model.T, domain=pyo.Binary)  # Binary variable for V_t sign

# Fix variables for jobs that a machine can't do
for i in I:
    for j in J:
        if j > n_jobs[i]:
            for t in T:
                model.x[i, t, j].fix(0)
                model.y[i, t, j].fix(0)


# Objective: Minimize battery and power costs plus deficit
def objective_rule(m):
    return m.N * c_b + m.M * c_p + sum(m.z[t] for t in m.T)
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


# Constraints

# 1. Energy balance constraint with deficit
def energy_balance_rule(m, t):
    return sum(e[i] * m.x[i, t, j] + f[i] * m.y[i, t, j] for i in m.I for j in m.J) - m.M * p[t] - m.s[t] <= m.z[t]
model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)


# 2. Volume calculation constraint
def volume_rule(m, t):
    return m.V[t] == sum(m.M * p[tp] - sum(e[i] * m.x[i, tp, j] + f[i] * m.y[i, tp, j] for i in m.I for j in m.J)
                         for tp in range(1, t))
model.volume_constraint = pyo.Constraint(model.T, rule=volume_rule)


# 3. Constraints for binary variable b_t based on V_t sign
def V_positive_rule(m, t):
    return m.V[t] >= -BIG_M * (1 - m.b[t])
model.V_positive = pyo.Constraint(model.T, rule=V_positive_rule)


def V_negative_rule(m, t):
    return m.V[t] <= BIG_M * m.b[t]
model.V_negative = pyo.Constraint(model.T, rule=V_negative_rule)


# 4. Storage constraints based on V_t and b_t
def storage_V_rule(m, t):
    return m.s[t] <= m.V[t] + BIG_M * (1 - m.b[t])
model.storage_V = pyo.Constraint(model.T, rule=storage_V_rule)


def storage_b_rule(m, t):
    return m.s[t] <= BIG_M * m.b[t]
model.storage_b = pyo.Constraint(model.T, rule=storage_b_rule)


# 5. Battery capacity constraint (it's already non negative)
def battery_capacity_rule(m, t):
    return m.s[t] <= m.N * B
model.battery_constraint = pyo.Constraint(model.T, rule=battery_capacity_rule)

# 6. Each job must run for required duration
model.job_requirements = pyo.ConstraintList()
for i in I:
    for j in J:
        if j <= n_jobs[i]:  # Only add constraint for required jobs
            model.job_requirements.add(
                sum(model.x[i, t, j] for t in T) == d[i]
            )


# 7. Each machine can do one job at a time
def one_job_at_time(m, i, t):
    return sum(m.x[i, t, j] for j in m.J) <= 1
model.single_job_time = pyo.Constraint(model.I, model.T, rule=one_job_at_time)


# 8. Each machine can start one job at a time
def one_start_at_time(m, i, t):
    return sum(m.y[i, t, j] for j in m.J) <= 1
model.single_start_time = pyo.Constraint(model.I, model.T, rule=one_start_at_time)


# 9. Max energy constraint
def max_energy(m, t):
    return sum(e[i] * m.x[i, t, j] for i in m.I for j in m.J) <= mmm[t]
model.max_energy = pyo.Constraint(model.T, rule=max_energy)

# 10. Silent periods for machines
model.silent_periods = pyo.ConstraintList()
for i, times in silent_periods.items():
    for t in times:
        if t <= T_MAX:  # Make sure the time is within our horizon
            model.silent_periods.add(sum(model.x[i, t, j] for j in J) == 0)

# 11. Shared resource constraint
model.shared_resources = pyo.ConstraintList()
for t in T:
    for group in M_shared:
        machines_in_group = [i for i in group if i in I]  # Ensure the machines are in our defined set
        if machines_in_group:
            model.shared_resources.add(
                sum(model.x[i, t, j] for i in machines_in_group for j in J) <= 1
            )


# 12. Start implies run and continuity constraint
def run_start_relation(m, i, t, j):
    if t == 1:
        # First time period: x[i,t,j] = y[i,t,j]
        return m.x[i, t, j] == m.y[i, t, j]
    else:
        # For subsequent periods: x[i,t,j] <= y[i,t,j] + x[i,t-1,j]
        return m.x[i, t, j] <= m.y[i, t, j] + m.x[i, t - 1, j]
model.run_start_relation = pyo.Constraint(model.I, model.T, model.J, rule=run_start_relation)


# 13. When you start a job, you must run it
def start_implies_run(m, i, t, j):
    return m.y[i, t, j] <= m.x[i, t, j]
model.start_implies_run = pyo.Constraint(model.I, model.T, model.J, rule=start_implies_run)

# 14. Dependency constraint
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

# 15. Cooldown constraint
model.cooldowns = pyo.ConstraintList()
for i in I:
    for t in T:
        if t > c[i]:  # Only apply for time periods after possible cooldown
            cooldown_sum = sum(1 - sum(model.x[i, tp, jj] for jj in J) for tp in range(t - c[i], t))
            for j in J:
                model.cooldowns.add(model.y[i, t, j] <= cooldown_sum / c[i])

# 16. Job must be completed before threshold
model.job_completion = pyo.ConstraintList()
for i in I:
    for j in J:
        if j <= n_jobs[i]:  # Only apply for required jobs
            # Ensure job runs for exact duration
            model.job_completion.add(
                sum(model.x[i, t, j] for t in T) == d[i]
            )
            # Ensure job is not scheduled after threshold
            for t in range(THRESHOLD_FOR_JOB_J_AND_I[(i, j)] + 1, T_MAX + 1):
                model.job_completion.add(model.x[i, t, j] == 0)

"""
# 17. Job continuity constraint
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
            for t in range(2, T_MAX + 1):
                # If the job is running at time t but not at t-1, then it must have started at t
                model.job_continuity.add(
                    model.x[i, t, j] - model.x[i, t - 1, j] <= model.y[i, t, j]
                )
"""

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

    # Print deficit values
    print("\nDeficit Values (z_t):")
    for t in range(1, min(11, T_MAX + 1)):  # Show first 10 time periods for brevity
        print(f"  t={t}: {pyo.value(model.z[t]):.2f}")

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

    # Print volume values
    print("\nVolume Values (V_t):")
    for t in range(1, min(11, T_MAX + 1)):  # Show first 10 time periods for brevity
        print(f"  t={t}: {pyo.value(model.V[t]):.2f}")

    # Create visualization of the schedule
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

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
    battery_capacity = int(pyo.value(model.N)) * B
    ax2.axhline(y=battery_capacity, color='r', linestyle='--', label=f'Battery Capacity ({battery_capacity})')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Energy Storage')
    ax2.set_title('Energy Storage Levels')
    ax2.legend()

    # Plot deficit
    deficit_values = [pyo.value(model.z[t]) for t in T]
    ax3.plot(T, deficit_values, marker='s', linestyle='-', markersize=4, color='red')
    ax3.set_xlabel('Time Period')
    ax3.set_ylabel('Energy Deficit')
    ax3.set_title('Energy Deficit (z_t)')

    plt.tight_layout()
    plt.savefig('schedule_visualization.png')
    print("\nSchedule visualization saved as 'schedule_visualization.png'")

    plt.show()
else:
    print("Failed to find an optimal solution.")