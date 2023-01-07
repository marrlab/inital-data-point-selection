

# delets elements from list l at indices
def delete_at_multiple_indices(l, indices):
    new_l = l.copy()
    for index in sorted(indices, reverse=True):
        del new_l[index]

    return new_l
