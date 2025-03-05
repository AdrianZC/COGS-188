import numpy as np


# Quantize the state space and action space
def quantize_state(state: dict, state_bins: dict) -> tuple:
    """
    
    Given the state and the bins for each state variable, quantize the state space.

    Args:
        state (dict): The state to be quantized.
        state_bins (dict): The bins used for quantizing each dimension of the state.

    Returns:
        tuple: The quantized representation of the state.
    """
    # TODO
    
    quantized = []

    for key, value in state.items():
        bin = state_bins[key]
        for i, dim_value in enumerate(value):
            min_val = bin[i][0]
            max_val = bin[i][-1]
            clipped_value = np.clip(dim_value, min_val, max_val)
            bin_index = np.digitize(clipped_value, bin[i]) - 1
            bin_index = max(0, min(bin_index, len(bin[i]) - 2))
            quantized.append(bin_index)

    return tuple(quantized)

def quantize_action(action: float, bins: list) -> int:
    """
    Quantize the action based on the provided bins. 
    """
    # TODO

    min_val = bins[0]
    max_val = bins[-1]
    clipped_action = np.clip(action, min_val, max_val)
    bin_index = np.digitize(clipped_action, bins) - 1
    bin_index = max(0, min(bin_index, len(bins) - 2))
    
    return bin_index