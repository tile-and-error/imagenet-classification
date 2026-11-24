import pytest
import torch
import torch.nn.functional as F

from lwfm.core.losses import LossBinding, LossComposer
from lwfm.techniques.distillation import DIST, DKD, KD, EmbedCos, EmbedMSE

generator = torch.Generator().manual_seed(0)
Z_S = torch.randn(8, 5, generator=generator)  # student logits
Z_T = torch.randn(8, 5, generator=generator)  # teacher logits
Y = torch.randint(0, 5, (8,), generator=generator)


@pytest.mark.parametrize(
    "loss",
    [lambda z: KD()(z, z), lambda z: DKD()(z, z, Y), lambda z: DIST()(z, z)],
    ids=["kd", "dkd", "dist"],
)
def test_identical_logits_give_zero_loss(loss):
    assert loss(Z_S).item() == pytest.approx(0.0, abs=1e-5)


def test_kd_is_t_squared_times_the_kl_of_softened_outputs():
    T = 4.0
    p_t = F.softmax(Z_T / T, 1)
    kl = (p_t * (p_t.log() - F.log_softmax(Z_S / T, 1))).sum(1).mean()
    assert KD(T)(Z_S, Z_T).item() == pytest.approx(T**2 * kl.item(), rel=1e-5)


def test_weighted_kd_with_equal_weights_is_plain_kd():
    weighted = KD(weight=torch.ones(5))(Z_S, Z_T, Y)
    torch.testing.assert_close(weighted, KD()(Z_S, Z_T))


def test_weighted_kd_weights_each_sample_by_its_class():
    # Only class 0's samples count when the other classes weigh 0
    weight = torch.tensor([1.0, 0, 0, 0, 0])
    labels = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
    only_class_0 = KD()(Z_S[:2], Z_T[:2])
    torch.testing.assert_close(KD(weight=weight)(Z_S, Z_T, labels), only_class_0)


def test_weighted_kd_needs_the_labels():
    with pytest.raises(ValueError, match="labels"):
        KD(weight=torch.ones(5))(Z_S, Z_T)


@pytest.mark.parametrize("loss_class", [DKD, DIST])
def test_the_two_terms_add_up(loss_class):
    # DKD(alpha, beta) and DIST(beta, gamma) are weighted sums of two terms
    first, second = loss_class(1.0, 0.0), loss_class(0.0, 1.0)
    args = (Z_S, Z_T, Y) if loss_class is DKD else (Z_S, Z_T)
    both = loss_class(1.0, 1.0)(*args)
    torch.testing.assert_close(both, first(*args) + second(*args))


def test_embed_cos_matches_direction_only():
    h = torch.randn(4, 6, generator=generator)
    assert EmbedCos()(3 * h, h).item() == pytest.approx(0.0, abs=1e-6)  # any norm
    assert EmbedCos()(h, -h).item() == pytest.approx(2.0)  # opposite


def test_embed_mse_standardises_the_teacher_embedding():
    h_s, h_t = (
        torch.randn(4, 3, generator=generator),
        torch.randn(4, 3, generator=generator),
    )
    mean, std = torch.tensor([1.0, 0, -1]), torch.tensor([2.0, 1, 0.5])
    expected = F.mse_loss(h_s, (h_t - mean) / std)
    torch.testing.assert_close(EmbedMSE(mean, std)(h_s, h_t), expected)


def test_buffers_move_with_the_loss_composer():
    # The class weights and statistics are buffers, so composer.to(device) moves them
    kd = KD(weight=torch.ones(5))
    mse = EmbedMSE(torch.zeros(3), torch.ones(3))
    composer = LossComposer(
        [
            LossBinding("kd", kd, ["s.logits", "t.logits", "targets.class"]),
            LossBinding("mse", mse, ["s.embed", "t.embed"]),
        ]
    )
    assert set(composer.state_dict()) == {
        "loss_modules.0.weight",
        "loss_modules.1.mean",
        "loss_modules.1.std",
    }
