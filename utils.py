def n_params(model):
    """Return the number of parameters in a pytorch model.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        int: The number of parameters in the model.
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
