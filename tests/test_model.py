import pytest

torch = pytest.importorskip("torch")

from vajra.model.losses import drop_modalities, focal_loss  # noqa: E402
from vajra.model.net import FiLM, VajraNet  # noqa: E402

GROUPS = {"radar": 1, "satellite": 2, "lightning": 1, "model": 2}


def test_forward_shape_and_lead_time_conditioning():
    torch.manual_seed(0)
    net = VajraNet(GROUPS, widths=(8, 16, 32))
    x = torch.randn(2, 3, 6, 32, 32)
    assert net(x, torch.tensor([30, 30])).shape == (2, 2, 32, 32)
    with torch.no_grad():  # lead-time layers start neutral; nudge them
        for m in net.modules():
            if isinstance(m, FiLM):
                m.net.weight.normal_(0, 0.1)
    assert not torch.allclose(net(x, torch.tensor([10, 10])), net(x, torch.tensor([120, 120])))


def test_missing_sensor_changes_the_input_not_the_shape():
    net = VajraNet(GROUPS, widths=(8, 16, 32))
    x = torch.randn(1, 2, 6, 32, 32)
    full = net(x, torch.tensor([60]))
    no_radar = net(x, torch.tensor([60]), torch.tensor([[0.0, 1, 1, 1]]))
    assert full.shape == no_radar.shape and not torch.allclose(full, no_radar)


def test_drop_modalities_always_keeps_one_group():
    out = drop_modalities(torch.ones(500, 4), p=0.9, generator=torch.Generator().manual_seed(0))
    assert (out.sum(1) >= 1).all() and out.mean() < 0.5


def test_focal_loss_prefers_correct_predictions():
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    good = focal_loss(torch.tensor([4.0, -4.0, -4.0, -4.0]), target)
    bad = focal_loss(torch.tensor([-4.0, 4.0, 4.0, 4.0]), target)
    assert good < bad


def test_training_steps_reduce_the_loss():
    torch.manual_seed(0)
    net = VajraNet(GROUPS, widths=(8, 16, 32))
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    x = torch.randn(2, 2, 6, 32, 32)
    y = (torch.rand(2, 2, 32, 32) > 0.9).float()
    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = focal_loss(net(x, torch.tensor([30, 60])), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
