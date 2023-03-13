import torch

from src.akita.models import shift_pad, reverse_complement, reverse_triu


def test_shift_seq():
    x = torch.tensor([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    ]).float()

    shift_x_l = torch.tensor([
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25]],
        [[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]]
    ])

    shift_x_r = torch.tensor([
        [[0.25, 0.25, 0.25, 0.25], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        [[0.25, 0.25, 0.25, 0.25], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    ])

    assert torch.equal(shift_x_l, shift_pad(x, shift=-1))
    assert torch.equal(shift_x_r, shift_pad(x, shift=1))


def test_reverse_compliment():
    x = torch.tensor([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    ]).float()

    rc_x = torch.tensor([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
    ]).float()

    assert torch.equal(rc_x, reverse_complement(x))


def test_reverse_triu():
    sym_mat = torch.randn(10, 10)
    sym_mat = sym_mat + sym_mat.T
    rev_sym_mat = torch.flip(sym_mat, dims=[0, 1])

    triu_idx = torch.triu_indices(10, 10, offset=2)
    triu = sym_mat[triu_idx[0], triu_idx[1]]
    rev_triu = rev_sym_mat[triu_idx[0], triu_idx[1]]

    triu_batch = torch.stack([triu] * 3, dim=0).unsqueeze(2)
    rev_triu_batch = torch.stack([rev_triu] * 3, dim=0).unsqueeze(2)
    assert torch.equal(rev_triu_batch, reverse_triu(triu_batch, 10, 2))


if __name__ == "__main__":
    test_shift_seq()
    test_reverse_compliment()
    test_reverse_triu()
    print("Done.")
