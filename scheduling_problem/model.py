import pyomo.environ as pyo
import random

# Model
model = pyo.ConcreteModel()

# Parameters (sample data generation)
n = 3  # machines
MAX_JOB_N = 2
T = list(range(1, 1441))  # 1 to 1440
I = list(range(1, n+1))
J = list(range(1, MAX_JOB_N+1))

model.T = pyo.Set(initialize=T)
model.I = pyo.Set(initialize=I)
model.J = pyo.Set(initialize=J)

e = {i: random.randint(1, 5) for i in I}
f = {i: random.randint(10, 20) for i in I}
p = {t: random.randint(0, 100) for t in T}
m = {t: 100 for t in T}
d = {i: random.randint(20, 100) for i in I}
n_jobs = {i: MAX_JOB_N for i in I}
c = {i: random.randint(10, 30) for i in I}

# Sets of dependencies and shared resources (example)
M_dependencies = [(1, 2)]
M_shared = [{1, 3}]

# Variables
model.x = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)
model.y = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)

# Objective: Minimize wasted energy
def objective_rule(m):
    return max(0,sum(e[i] * m.x[i, t, j] + f[i] * m.y[i, t, j] - p[t]
               for t in m.T for i in m.I for j in m.J))
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Constraints

# 1. Total usage >= needed
def usage_requirement(m, i):
    return sum(m.x[i, t, j] for t in m.T for j in m.J) >= n_jobs[i] * d[i]
model.usage_req = pyo.Constraint(model.I, rule=usage_requirement)

# 2. Each machine can do one job at a time
def one_job_at_time(m, i, t):
    return sum(m.x[i, t, j] for j in m.J) <= 1
model.single_job_time = pyo.Constraint(model.I, model.T, rule=one_job_at_time)

# 3. Same for starting
def one_start_at_time(m, i, t):
    return sum(m.y[i, t, j] for j in m.J) <= 1
model.single_start_time = pyo.Constraint(model.I, model.T, rule=one_start_at_time)

# 4. Max energy constraint
def max_energy(m, t):
    return sum(e[i] * m.x[i, t, j] for i in m.I for j in m.J) <= m[t]
#model.max_energy = pyo.Constraint(model.T, rule=max_energy)

#TODO: put the silent orary constraint

# 5. Shared resource constraint
def shared_resources(m, t, g):
    return sum(m.x[i, t, j] for i in g for j in m.J if i in m.I) <= 1
model.shared_resources = pyo.ConstraintList()
for t in T:
    for group in M_shared:
        model.shared_resources.add(shared_resources(model, t, group))

# 6. Start implies run
def start_implies_run(m, i, t, j):
    return m.y[i, t, j] <= m.x[i, t, j]
model.start_condition = pyo.Constraint(model.I, model.T, model.J, rule=start_implies_run)

# 6. Start implies run
def start_implies_run_2(m, i, t, j):
    if t == 1:
        return m.y[i, t, j] <= m.x[i, t, j]
    return m.x[i, t, j] <= m.y[i, t, j] + m.x[i, t-1, j]
model.start_condition_2 = pyo.Constraint(model.I, model.T, model.J, rule=start_implies_run_2)


# 7. Dependency constraint
def dependency_rule(m, k, kp1, t, j):
    return m.y[kp1, t, j] <= sum(m.x[k, tp, j] for tp in m.T if tp < t) / d[k]
model.dependencies = pyo.ConstraintList()
for (k, kp1) in M_dependencies:
    for t in T:
        for j in J:
            model.dependencies.add(dependency_rule(model, k, kp1, t, j))

# 8. Cooldown constraint
def cooldown_rule(m, i, t, j):
    start = max(1, t - c[i])
    return m.y[i, t, j] <= sum(1 - m.x[i, tp, j] for tp in range(start, t)) / c[i]
model.cooldowns = pyo.Constraint(model.I, model.T, model.J, rule=cooldown_rule)

# 9. Job duration enforcement TODO: only for some time t
def duration_rule(m, i, j):
    return sum(m.x[i, t, j] for t in m.T) == d[i]
model.job_duration = pyo.Constraint(model.I, model.J, rule=duration_rule)

# Solve it
solver = pyo.SolverFactory('glpk')
result = solver.solve(model)

# Print status and basic solution
print(f"Status: {result.solver.status}, Termination: {result.solver.termination_condition}")
for i in I:
    for j in J:
        start_times = [t for t in T if pyo.value(model.y[i, t, j]) > 0.5]
        if start_times:
            print(f"Machine {i} Job {j} starts at: {start_times}")
