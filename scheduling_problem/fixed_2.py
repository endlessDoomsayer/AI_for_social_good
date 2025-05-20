
N = 5 # number of batteries output of phase 1
M = 10 # number of panels output of phase 1
c_b = 1000000  # Cost per unit of battery capacity
c_p = 500000   # Cost per unit of power capacity

# Maximum battery capacity
B = 100

total_cost_panels = N*c_b + M*c_p

# Sets
T = list(range(1, 11))  # Time periods
I = list(range(1, 6))   # Machines
J = list(range(1, 4))   # Jobs

# Info on machines per day
e = {i: 10000 + i*2 for i in I}  # Energy consumption rate for machine i
f = {i: 5000 + i for i in I}     # Fixed energy cost for machine i
n = {i: i for i in I}         # Minimum number of jobs required for machine i
d = {i: 13+i for i in I}         # Duration required for machine i to complete a job

total_energy_per_day = sum(e[i]*d[i]*n[i]+f[i]*n[i] for i in I)

# Cost of energy if taken from outside
c_e = 10
total_cost_per_day = (c_e*total_energy_per_day)

# Total days we could go on without panels
days = total_cost_panels/total_cost_per_day
print("Days if we don't buy panels:", days)
print("Years if we don't buy panels:", days/365)
