import torch
from torch import nn

from lerobot.utils.training_performance import (
    defer_scalar_metrics, materialize_scalar_metrics, prepare_isolated_grad_clip_cache, scalar_metric,
)


def test_metrics_preserve_values_gradients_rng_and_context():
    x = torch.tensor(1.2345678901234567, dtype=torch.float64, requires_grad=True)
    rng = torch.get_rng_state().clone()
    with defer_scalar_metrics():
        values = {'double': scalar_metric(x), 'float': scalar_metric(x.float()), 'constant': 3.0}
        assert isinstance(values['double'], torch.Tensor) and not values['double'].requires_grad
        with defer_scalar_metrics(False):
            assert scalar_metric(x) == x.item()
    actual = materialize_scalar_metrics(values)
    assert actual == {'double': x.item(), 'float': x.float().item(), 'constant': 3.0}
    assert scalar_metric(x) == x.item()
    assert torch.equal(rng, torch.get_rng_state())
    (x * x).backward()
    assert x.grad.item() == (2 * x).item()


def test_cached_clipping_preserves_parameter_order_and_optimizer_update():
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Linear(2, 2)
            self.term = nn.Linear(2, 1)
            self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)
            self.calls = 0

        def isolated_main_optimizer_grad_groups(self):
            self.calls += 1
            return {'term': list(self.term.parameters())}

    torch.manual_seed(9)
    baseline, cached = Policy(), Policy()
    cached.load_state_dict(baseline.state_dict())
    main, groups = prepare_isolated_grad_clip_cache(cached)
    assert list(main) == list(cached.main.parameters())
    for model in [baseline, cached]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            (model.main(torch.ones(2, 2)).square().mean() + model.term(torch.ones(2, 2)).square().mean()).backward()
            if model is baseline:
                group = model.isolated_main_optimizer_grad_groups()['term']
                excluded = {id(p) for p in group}
                ordinary = [p for p in model.parameters() if p.requires_grad and id(p) not in excluded]
            else:
                ordinary, group = main, groups['term']
            torch.nn.utils.clip_grad_norm_(ordinary, .1)
            torch.nn.utils.clip_grad_norm_(group, .1)
            optimizer.step()
    assert cached.calls == 1 and baseline.calls == 3
    for a, b in zip(baseline.parameters(), cached.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
