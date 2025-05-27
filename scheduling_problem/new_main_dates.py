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
import scheduling_problem.run_models_dates as run_models_dates

# TODO attento verifica che in combine data utilizzi la data corretta anche per i pannelli solari

#MODEL_1_POLICIES = [ '_1_bin_search_improved_backtracking', '_1_bin_search_scip', '_1_glpk')
MODEL_1_POLICIES = [ '_1_bin_search_scip']
MODEL_3_POLICIES = [ '_3_local_search', '_3_milp']
MODEL_4_1_POLICIES = { '_4_1_enhanced': (500, 330), '_4_1_lin_prog': (500, 330), '_4_1_milp': (500, 330),'_4_1_lin_prog': (500, 330)}
MODEL_4_3_POLICIES = { '_4_3_milp': (500, 330)}

# Run Model 1
number_of_M_N_per_policy = run_models_dates.run_models_1(MODEL_1_POLICIES, days=7)
current_M, current_N = 0, 0
current_date = "2018-01-01"
for policy, (M, N, date) in number_of_M_N_per_policy.items():
    current_M = max(current_M, M)
    current_N = max(current_N, N)
    current_date = max(current_date, date)

# Run step 2
days, years = run_models_dates.run_step_2(MODEL_1_POLICIES)
number_of_cost_M_N_per_policy = run_models_dates.run_models_3(MODEL_3_POLICIES, days=days)

run_models_dates.run_models_4(MODEL_4_1_POLICIES, days=1)
run_models_dates.run_models_4(MODEL_4_3_POLICIES, days=1)

