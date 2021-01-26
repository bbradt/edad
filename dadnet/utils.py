def n_bits(tensor):
    if tensor is None:
        return 0
    return 8 * tensor.nelement() * tensor.element_size()
