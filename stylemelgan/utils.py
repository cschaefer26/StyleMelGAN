from torch.nn import Module
from torch.nn.utils import remove_weight_norm


def get_children(model: Module):
    children = list(model.children())
    flat_children = []
    if not children:
        return model
    else:
        for child in children:
            try:
                flat_children.extend(get_children(child))
            except TypeError:
                flat_children.append(get_children(child))
    return flat_children


def remove_weight_norm_recursively(model: Module) -> None:
    layers = get_children(model)
    for l in layers:
        try:
            remove_weight_norm(l)
            print(l)
        except Exception as e:
            pass