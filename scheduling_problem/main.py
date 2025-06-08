
import run_models as run_models

MODEL_1_POLICIES = [ '_1_scip']
MODEL_3_POLICIES = [ '_3_scip']
MODEL_4_1_POLICIES = { '_4_1_scip': (0, 0)}
MODEL_4_3_POLICIES = { '_4_3_scip': (0, 0)}

days = 2
date = "2018-02-19"

# Run Model 1
model_1_M, model_1_N = 999999999, 999999999
number_of_M_N_per_policy = run_models.run_models_1(MODEL_1_POLICIES, days=days, date=date)
for policy, (M, N) in number_of_M_N_per_policy.items():
    model_1_M, model_1_N = min(model_1_M, M), min(model_1_N, N) # taking the minimum M and N across all policies


# Run step 2
number_of_days_years_per_policy = run_models.run_step_2(number_of_M_N=number_of_M_N_per_policy, policies=MODEL_1_POLICIES, days=days, date=date)

step_2_days = 999999999
for policy, (days, years) in number_of_days_years_per_policy.items():
    step_2_days = min(step_2_days, days)  

number_of_cost_M_N_per_policy = run_models.run_models_3(policies=MODEL_3_POLICIES, days=days, date=date, tot_number_of_days = step_2_days)
model_3_cost, model_3_M, model_3_N = 999999999, 999999999, 999999999
for policy, (cost, M, N) in number_of_cost_M_N_per_policy.items():
    model_3_cost, model_3_M, model_3_N = min(model_3_cost, cost), min(model_3_M, M), min(model_3_N, N)

# Run Model 4
for policy, (M, N) in MODEL_4_1_POLICIES.items():
    MODEL_4_1_POLICIES[policy] = (model_1_M, model_1_N)
for policy, (M, N) in MODEL_4_3_POLICIES.items():
    MODEL_4_3_POLICIES[policy] = (model_3_M, model_3_N)

run_models.run_models_4(number_of_M_N_per_policy=MODEL_4_1_POLICIES, days=2, date=date)
run_models.run_models_4(number_of_M_N_per_policy=MODEL_4_3_POLICIES, days=2, date=date)

