import pickle
import numpy as np 

model_path = "weather_pv_conversion/output/model/best_estimator_model.pkl"

try:
    with open(model_path, "rb") as f:
        loaded_object = pickle.load(f)

    print(f"Type of loaded object: {type(loaded_object)}")
    """
    if isinstance(loaded_object, np.ndarray):
        print(f"Shape of loaded NumPy array: {loaded_object.shape}")
        print("First few elements (if 1D or 2D):")
        if loaded_object.ndim == 1:
            print(loaded_object)
        elif loaded_object.ndim == 2:
            print(loaded_object[:5, :5]) # Print a small corner
        else:
            print("Array has more than 2 dimensions.")"""
    if hasattr(loaded_object, 'predict'):
        print("Object appears to be a model (has a 'predict' method).")
        # You could try to inspect further if it's a known model type
        # e.g., from sklearn, tensorflow, pytorch, etc.
    else:
        print("Loaded object is neither a NumPy array nor a recognizable model object.")
        print("Object content (be careful if it's very large):")
        print(loaded_object) # Uncomment with caution

except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
except ModuleNotFoundError as e:
    print(f"Error: A module required by the pickled object is missing: {e}")
    print("Please ensure all necessary libraries (e.g., sklearn, pandas, etc.) used to create the model are installed.")
except Exception as e:
    print(f"An error occurred while loading or inspecting the pickle file: {e}")