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


print("-----------------------------------\nModel 1\n-----------------------------------")
start = time.time()
_1_bin_search_improved_backtracking.solve()
end = time.time()

print(f"Time taken for binary search with improved backtracking: {end - start}")

start = time.time()
_1_bin_search_scip.solve()
end = time.time()

print(f"Time taken for binary search with SCIP: {end - start}")

start = time.time()
M,N = _1_glpk.solve()
end = time.time()

print(f"Time taken for GLPK: {end - start}")

print("-----------------------------------\nModel 2\n-----------------------------------")

_2.print_sol(M,N)


