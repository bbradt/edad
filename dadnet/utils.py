import copy


def n_bits(tensor):
    if tensor is None:
        return 0
    return 8 * tensor.nelement() * tensor.element_size()


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def get_average_model(surrogate_model, *args):
    surrogate_model.load_state_dict(args[0].state_dict())
    for model in args[1:]:
        state_dict = surrogate_model.state_dict()
        other_state_dict = model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = (v + other_state_dict[k]) / 2
        surrogate_model.load_state_dict(state_dict)
    return surrogate_model
