import torch


def test_checkpoint_restores_optimizer_scheduler_and_rng(tmp_path):
    from src.asr.runtime import save_checkpoint, restore_checkpoint
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    dest = tmp_path / "last.pt"
    save_checkpoint(dest, model, optimizer, scheduler, {"step": 1, "epoch": 2}, {"model": {}})
    for parameter in model.parameters():
        parameter.data.zero_()
    state = restore_checkpoint(dest, model, optimizer, scheduler)
    assert state == {"step": 1, "epoch": 2}
    assert all(torch.equal(model.state_dict()[k], v) for k, v in expected.items())
    assert scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == .005


def test_resume_next_step_with_partial_accumulation_and_dropout(tmp_path):
    from src.asr.runtime import save_checkpoint, restore_checkpoint
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(.4), torch.nn.Linear(8, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
    inputs = torch.ones(4, 3)
    (model(inputs).square().mean() / 2).backward()
    checkpoint = tmp_path / 'partial.pt'
    save_checkpoint(checkpoint, model, optimizer, scheduler,
                    {'step': 0, 'micro_count': 1}, {'model': {}})
    (model(inputs).square().mean() / 2).backward()
    optimizer.step()
    scheduler.step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    next_random = torch.rand(3)
    torch.rand(100)  # Simulate unrelated startup work before restoration.
    restore_checkpoint(checkpoint, model, optimizer, scheduler)
    (model(inputs).square().mean() / 2).backward()
    optimizer.step()
    scheduler.step()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected.items())
    assert torch.equal(next_random, torch.rand(3))
