from pyomo.environ import *
import random

# Data setup
n = 3
MAX_JOB_N = 2
T = list(range(1, 61))  # shorter horizon for testing
I = list(range(1, n+1))
J = list(range(1, MAX_JOB_N+1))

# Parameters
e = {i: random.randint(1, 5) for i in I}
f = {i: random.randint(10, 20) for i in I}
p = {t: random.randint(40, 100) for t in T}
m = {t: 100 for t in T}
d = {i: random.randint(5, 10) for i in I}
n_jobs = {i: MAX_JOB_N for i in I}
c = {i: random.randint(2, 5) for i in I}
M_dependencies = [(1, 2)]
M_shared = [{1, 3}]
B = 100  # capacity per battery
N = 3    # number of batteries

# Model
model = ConcreteModel()
model.T = Set(initialize=T)
model.I = Set(initialize=I)
model.J = Set(initialize=J)
model.x = Var(model.I, model.T, model.J, domain=Binary)
model.y = Var(model.I, model.T, model.J, domain=Binary)

# Objective (could be a dummy, model is feasibility-based)
model.obj = Objective(expr=0, sense=minimize)

# Battery feasibility: total excess should be ≤ battery capacity NB at every t
def battery_capacity_rule(m, t):
    return sum(p[tp] - e[i]*m.x[i, tp, j] - f[i]*m.y[i, tp, j]
               for tp in m.T if tp < t
               for i in m.I for j in m.J) <= N * B
model.battery_storage = Constraint(model.T, rule=battery_capacity_rule)

# Energy balance (should not draw more than available)
def net_energy_rule(m):
    return sum(e[i]*m.x[i, t, j] + f[i]*m.y[i, t, j] - p[t]
               for t in m.T for i in m.I for j in m.J) <= 0
model.net_energy = Constraint(rule=net_energy_rule)


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

# Solve
solver = SolverFactory('glpk')
result = solver.solve(model, tee=True)

# Output results
print(f"Status: {result.solver.status}, {result.solver.termination_condition}")
