import time
import _1_bin_search_improved_backtracking
import _1_bin_search_scip
import _1_glpk
import _2
import _3_local_search
import _3_milp
import _4_1_milp
import _4_1_enhanced
import _4_1_lin_prog
import _4_3_milp
import run_models

#MODEL_1_POLICIES = [ '_1_bin_search_improved_backtracking', '_1_bin_search_scip', '_1_glpk')
MODEL_1_POLICIES = [ '_1_bin_search_scip']
#MODEL_3_POLICIES = [ '_3_local_search', '_3_milp']
MODEL_3_POLICIES = [ '_3_local_search']
MODEL_4_1_POLICIES = { '_4_1_enhanced': (0, 0), '_4_1_lin_prog': (0, 0)}
MODEL_4_3_POLICIES = { '_4_3_milp': (0, 0)}


# Run Model 1
model_1_M, model_1_N = 999999999, 999999999
number_of_M_N_per_policy = run_models.run_models_1(MODEL_1_POLICIES, days=7, date="2018-01-01")
for policy, (M, N) in number_of_M_N_per_policy.items():
    model_1_M, model_1_N = min(model_1_M, M), min(model_1_N, N) # taking the minimum M and N across all policies


# Run step 2
step_2_days, step_2_years = run_models.run_step_2(number_of_M_N_per_policy, MODEL_1_POLICIES)


# Run Model 3
number_of_cost_M_N_per_policy = run_models.run_models_3(MODEL_3_POLICIES, tot_number_of_days = step_2_days)
model_3_cost, model_3_M, model_3_N = 999999999, 999999999, 999999999
for policy, (cost, M, N) in number_of_cost_M_N_per_policy.items():
    model_3_cost, model_3_M, model_3_N = min(model_3_cost, cost), min(model_3_M, M), min(model_3_N, N)


# Run Model 4
for policy, (M, N) in MODEL_4_1_POLICIES.items():
    MODEL_4_1_POLICIES[policy] = (model_1_M, model_1_N)
for policy, (M, N) in MODEL_4_3_POLICIES.items():
    MODEL_4_3_POLICIES[policy] = (model_3_M, model_3_N)

run_models.run_models_4(MODEL_4_1_POLICIES, days=1)
run_models.run_models_4(MODEL_4_3_POLICIES, days=1)

