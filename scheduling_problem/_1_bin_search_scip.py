from combine_data import get_data
from _4_1_scip import *
import time

def binary_search_N(min_M, max_N, data, is_feasible):
    """
    For a fixed M, binary search to find the minimum feasible N
    """
    low, high = 0, max_N
    result = None
    while low <= high:
        mid = (low + high) // 2
        if is_feasible(min_M, mid, data):
            result = mid
            high = mid - 1
        else:
            low = mid + 1
    return result

def binary_search_M(max_M, max_N, data, is_feasible):
    """
    Binary search to find the minimum feasible M, and for each M,
    binary search to find the minimum N.
    """
    low, high = 0, max_M
    best_MN = None

    while low <= high:
        mid_M = (low + high) // 2
        min_N = binary_search_N(mid_M, max_N, data, is_feasible)
        if min_N is not None:
            best_MN = (mid_M, min_N)
            high = mid_M - 1  # Try to minimize M further
        else:
            low = mid_M + 1

    return best_MN

def solve(data = get_data()):
    # Define the search bounds for M and N
    biggestM = 10000
    biggestN = 10000
    
    start = time.time()
    result = binary_search_M(biggestM, biggestN, data, find_min)
    end = time.time()
    
    print(f"Time taken for binary search with SCIP: {end - start}")
    
    if result:
        print(f"Minimum feasible (M, N): {result}")
    else:
        print("No feasible (M, N) found in given bounds.")
    print_solution(result[0],result[1],data,"1_bin_search_scip")
    return result
    
if __name__ == "__main__":
    solve()