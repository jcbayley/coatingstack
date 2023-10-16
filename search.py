import numpy as np

import numpy as np
import copy

MATERIAL_LIST = {
    "m1": {"n": 2},
    "m2": {"n": 8},
    "m3": {"n": 6}
}

def compute_loss(material_order):
    """_summary_

    Args:
        material_order (_type_): _description_

    Returns:
        _type_: _description_
    """
    loss = 0
    prev_material = None
    for i, material in enumerate(material_order):
        if i == 0:
            prev_material = material
        else:
            refractive_index = np.abs(material["n"] - prev_material["n"])
            loss += refractive_index
            prev_material = material


    return -loss


def make_layers(n_layers):
    """_summary_

    Args:
        n_layers (_type_): _description_

    Returns:
        _type_: _description_
    """
    layer_names = list(MATERIAL_LIST.keys())

    material_order = []
    for i in range(n_layers):
        val = np.random.choice(3)
        material_order.append(copy.copy(MATERIAL_LIST[layer_names[val]]))

    return material_order


def run_random_search(n_iterations, n_layers):
    """_summary_

    Args:
        n_iterations (_type_): _description_
        n_layers (_type_): _description_

    Returns:
        _type_: _description_
    """
    final_loss = np.inf
    final_layers = None
    for i in range(n_iterations):
        layer_order = make_layers(n_layers)

        loss = compute_loss(layer_order)
        print(loss, layer_order)
        if loss < final_loss:
            final_loss = loss
            final_layers = layer_order

        
        
    return final_loss, final_layers


if __name__ == "__main__":

    loss, fl = run_random_search(100, 10)
    print("final", loss, fl)