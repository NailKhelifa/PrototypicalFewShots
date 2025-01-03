import torch


def get_n_per_classes(x_enroll, y_enroll, n=5):

    classes = torch.unique(y_enroll)
    idxs = []

    for label in classes:

        lab_idx = (y_enroll == label).nonzero(as_tuple=True)[0]

        if len(lab_idx) > n:
            idx = lab_idx[torch.randperm(len(lab_idx))[:n]]
        else:

            idx = lab_idx

        idxs.append(idx)

    idxs = torch.cat(idxs)

    x_n = x_enroll[idxs]
    y_n = y_enroll[idxs]
    return (x_n, y_n), idxs
