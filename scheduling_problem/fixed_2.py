from combine_data import get_data

N = 5 # number of batteries output of phase 1
M = 10 # number of panels output of phase 1
c_e = 10 # cost of energy if taken from outside

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
B = data["B"]

# Calculations

total_cost_panels = N*c_b + M*c_p

total_energy_per_day = sum(e[i]*d[i]*n_jobs[i]+f[i]*n_jobs[i] for i in I)

total_cost_per_day = (c_e*total_energy_per_day)

# Total days we could go on without panels
days = total_cost_panels/total_cost_per_day
print("Days if we don't buy panels:", days)
print("Years if we don't buy panels:", days/365)
