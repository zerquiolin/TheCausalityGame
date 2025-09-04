import numpy as np
import json


def random_state_to_json(rs):
    rng_state = rs.get_state()
    state_dict = {
        "bit_generator": "RandomState",
        "state": {
            "key": rng_state[1].tolist(),  # numpy array to list
            "pos": rng_state[2],
            "has_gauss": rng_state[3],
            "cached_gaussian": rng_state[4],
            "state_name": rng_state[0],  # usually 'MT19937'
        },
    }
    return json.dumps(state_dict)


def random_state_from_json(json_str):
    loaded_dict = json.loads(json_str)
    restored_rng = np.random.RandomState()
    restored_rng.set_state(
        (
            loaded_dict["state"]["state_name"],
            np.array(loaded_dict["state"]["key"], dtype=np.uint32),
            loaded_dict["state"]["pos"],
            loaded_dict["state"]["has_gauss"],
            loaded_dict["state"]["cached_gaussian"],
        )
    )
    return restored_rng


def random_state_from_dict(state_dict):
    restored_rng = np.random.RandomState()
    restored_rng.set_state(
        (
            state_dict["state"]["state_name"],
            np.array(state_dict["state"]["key"], dtype=np.uint32),
            state_dict["state"]["pos"],
            state_dict["state"]["has_gauss"],
            state_dict["state"]["cached_gaussian"],
        )
    )
    return restored_rng
