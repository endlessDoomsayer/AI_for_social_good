
import run_models 

MODEL_1_POLICIES = [ '_1_glpk']

# Run Model 1
number_of_M_N_per_policy = run_models.run_models_1(MODEL_1_POLICIES, days=7, date="2018-01-01")
print(f"Model 1 results: {number_of_M_N_per_policy}")