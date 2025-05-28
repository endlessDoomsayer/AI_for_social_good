from combine_data import get_data
from _4_1_lin_prog import *
import time

def binary_search_M(min_N, max_M, data, is_feasible):
    """
    For a fixed M, binary search to find the minimum feasible N
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

def binary_search_N(max_M, max_N, data, is_feasible):
    """
    Binary search to find the minimum feasible M, and for each M,
    binary search to find the minimum N.
    """
    low, high = 0, max_N
    best_MN = None

    while low <= high:
        mid_N = (low + high) // 2
        min_M = binary_search_M(mid_N, max_M, data, is_feasible)
        if min_M is not None:
            best_MN = (mid_N, min_M)
            high = mid_N - 1  # Try to minimize M further
        else:
            low = mid_N + 1

    return best_MN

def solve(biggestM = 5000, biggestN = 1000, data = get_data()):

    start = time.time()
    result = binary_search_N(biggestM, biggestN, data, find_min)
    end = time.time()
    
    print(f"Time taken for binary search with SCIP: {end - start}")
    
    if result:
        print(f"Minimum feasible (M, N): {result}")
    else:
        print("No feasible (M, N) found in given bounds.")
    print_solution(result[0],result[1],data,"1_scip")
    
    return result[0],result[1]

if __name__ == "__main__":
    solve()