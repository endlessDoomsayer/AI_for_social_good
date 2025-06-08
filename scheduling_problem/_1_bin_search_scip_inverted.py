from combine_data import get_data
from _4_1_scip import *
import time

def binary_search_M(min_N, max_M, data, is_feasible):
    """
    For a fixed N, binary search to find the minimum feasible M
    """
    low, high = 0, max_M
    result = None
    while low <= high:
        mid = (low + high) // 2
        if is_feasible(mid, min_N, data):
            result = mid
            high = mid - 1
        else:
            low = mid + 1
    return result

def binary_search_N(max_N, max_M, data, is_feasible):
    """
    Binary search to find the minimum feasible N, and for each N,
    binary search to find the minimum M.
    """
    low, high = 0, max_N
    best_NM = None

    while low <= high:
        mid_N = (low + high) // 2
        min_M = binary_search_M(mid_N, max_M, data, is_feasible)
        if min_M is not None:
            best_NM = (mid_N, min_M)
            high = mid_N - 1  # Try to minimize N further
        else:
            low = mid_N + 1

    return best_NM

def solve(data = get_data()):
    # Define the search bounds for M and N

    biggestM = 10000
    biggestN = 10000

   
    
    
    start = time.time()
    result = binary_search_N(biggestN, biggestM, data, find_min)
    end = time.time()
    
    print(f"Time taken for binary search with SCIP: {end - start}")
    
    if result:
        print(f"Minimum feasible (N, M): {result}")
    else:
        print("No feasible (N, M) found in given bounds.")
    print_solution(result[1],result[0],data,"1_bin_search_scip_inverted")
    return result[1], result[0]
    
if __name__ == "__main__":
    solve()