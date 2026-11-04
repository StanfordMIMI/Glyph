import torch
import pytest

from glyph.config import load_config
from glyph.mechinterp.sae import TopKSAE
from glyph.mechinterp.sae_trainer import calibrate_encoder_initialization, train_sae


def test_training_gain_reduces_initial_error_and_preserves_topk_support():
    torch.manual_seed(17)
    x = torch.randn(61, 8)
    model = TopKSAE(8, 128, 32)
    encoder = model.W_enc.detach().clone()
    decoder = model.W_dec.detach().clone()
    with torch.no_grad():
        original = model(x)
        support = original['z_sparse'] != 0
        before = ((original['x_hat'] - x) ** 2).mean().item()
    diagnostic = calibrate_encoder_initialization(model, x, [128], batch_size=11)
    with torch.no_grad():
        calibrated = model(x)
        after = ((calibrated['x_hat'] - x) ** 2).mean().item()
    assert 0 < diagnostic['scale'] < 1
    assert after < before
    assert after == pytest.approx(diagnostic['mse_after'], rel=1e-5, abs=1e-7)
    assert torch.equal(calibrated['z_sparse'] != 0, support)
    torch.testing.assert_close(model.W_dec, decoder, rtol=0, atol=0)
    torch.testing.assert_close(model.W_enc, encoder * diagnostic['scale'])
    torch.testing.assert_close(calibrated['x_hat'], original['x_hat'] * diagnostic['scale'])


def test_nested_dictionary_gain_reduces_the_actual_prefix_loss():
    torch.manual_seed(21)
    x = torch.randn(35, 6)
    model = TopKSAE(6, 96, 24)
    def loss():
        with torch.no_grad():
            sparse = model(x)['z_sparse']
            return sum(((model.reconstruct_with_prefix(sparse, n) - x) ** 2).mean().item()
                       for n in [16, 48, 96]) / 3
    before = loss()
    diagnostic = calibrate_encoder_initialization(model, x, [16, 48, 96])
    assert loss() <= before
    assert loss() == pytest.approx(diagnostic['mse_after'], rel=1e-5)


def test_zero_training_inputs_do_not_create_nan_weights():
    model = TopKSAE(4, 16, 4)
    diagnostic = calibrate_encoder_initialization(model, torch.zeros(12, 4))
    assert diagnostic == dict(scale=1., mse_before=0., mse_after=0.)
    assert torch.isfinite(model.W_enc).all()


def test_legacy_initialization_opt_out_does_not_calibrate(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Legacy opt-out called the new calibration')
    monkeypatch.setattr('glyph.mechinterp.sae_trainer.calibrate_encoder_initialization', forbidden)
    cfg = load_config(overrides=['sae.matryoshka=[8]', 'sae.top_ks=[2]', 'sae.max_steps=1',
        'sae.calibrate_encoder_init=false', 'sae.use_ghost_grads=false'])
    model, _ = train_sae(torch.randn(12, 4), torch.randn(6, 4), 4, 8, 2, cfg, device='cpu')
    assert model.encoder_initialization == 'legacy_transpose'
    assert model.encoder_initialization_scale == 1.
