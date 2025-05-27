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

# TODO attento verifica che in combine data utilizzi la data corretta anche per i pannelli solari

#MODEL_1_POLICIES = [ '_1_bin_search_improved_backtracking', '_1_bin_search_scip', '_1_glpk')
MODEL_1_POLICIES = [ '_1_bin_search_scip']
MODEL_3_POLICIES = [ '_3_local_search', '_3_milp']
MODEL_4_1_POLICIES = { '_4_1_enhanced': (500, 330), '_4_1_lin_prog': (500, 330), '_4_1_milp': (500, 330),'_4_1_lin_prog': (500, 330)}
MODEL_4_3_POLICIES = { '_4_3_milp': (500, 330)}

'''
number_of_M_N_per_policy = run_models.run_models_1(MODEL_1_POLICIES, days=7)
days, years = run_models.run_step_2(MODEL_1_POLICIES)
number_of_cost_M_N_per_policy = run_models.run_models_3(MODEL_3_POLICIES, days=days)
'''
run_models.run_models_4(MODEL_4_1_POLICIES, days=1)
run_models.run_models_4(MODEL_4_3_POLICIES, days=1)

