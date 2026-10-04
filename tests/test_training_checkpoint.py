import random
import numpy as np
import pytest


def test_atomic_resume_restores_update_rng_and_sampler(tmp_path):
    torch = pytest.importorskip('torch')
    from emma_gpm.training_checkpoint import save_checkpoint, restore_checkpoint
    from scripts.pretrain_videomae import build_loader
    from argparse import Namespace

    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    dataset = [{'pixel_values': torch.full((3,), float(i))} for i in range(8)]
    args = Namespace(batch_size=4, num_workers=0, prefetch_factor=1, pin_memory=False, seed=42)
    loader = build_loader(dataset, args, torch.device('cpu'))
    generators = {'loader': loader.generator, 'sampler': loader.sampler.generator}
    def epoch(current_loader):
        order=[]
        for b in current_loader:
            x=b['pixel_values'];order.extend(x[:,0].tolist())
            optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
        scheduler.step()
        return order
    epoch(loader)
    path=tmp_path/'checkpoint.pt'
    identity={'dataset':'test'}
    save_checkpoint(path, model, optimizer, scheduler, generators, {'completed_epochs':1}, identity)
    expected_random=(random.random(), np.random.random(), torch.rand(3))
    expected_order=epoch(loader)
    expected_parameters=[p.detach().clone() for p in model.parameters()]
    restarted=build_loader(dataset,args,torch.device('cpu'))
    progress=restore_checkpoint(path,model,optimizer,scheduler,
                                {'loader':restarted.generator,'sampler':restarted.sampler.generator},identity)
    assert progress['completed_epochs']==1
    assert random.random()==expected_random[0] and np.random.random()==expected_random[1]
    assert torch.equal(torch.rand(3),expected_random[2])
    assert epoch(restarted)==expected_order
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),expected_parameters))
    assert not path.with_suffix('.pt.part').exists()
    with pytest.raises(ValueError,match='differs'):
        restore_checkpoint(path,model,optimizer,scheduler,generators,{'dataset':'different'})


def test_cuda_checkpoint_restores_tube_mask_rng(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from types import SimpleNamespace
    from emma_gpm.training_checkpoint import save_checkpoint, restore_checkpoint
    from scripts.pretrain_videomae import make_tube_mask
    model = torch.nn.Linear(2, 2).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    generators = {'sampler': torch.Generator().manual_seed(42)}
    config = SimpleNamespace(image_size=224, patch_size=16, num_frames=16, tubelet_size=2)
    torch.cuda.manual_seed_all(42)
    path = tmp_path / 'cuda_checkpoint.pt'
    save_checkpoint(path, model, optimizer, scheduler, generators, {}, {'device':'cuda'})
    expected = make_tube_mask(config, 2, .9, 'cuda')
    make_tube_mask(config, 2, .9, 'cuda')
    restore_checkpoint(path, model, optimizer, scheduler, generators, {'device':'cuda'})
    assert torch.equal(expected, make_tube_mask(config, 2, .9, 'cuda'))
