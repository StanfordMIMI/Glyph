from types import SimpleNamespace
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from glyph.models.huggingface import HFEmbeddingModel
from glyph.models.registry import get_model


def pooling(mode='auto', key=None):
    model = HFEmbeddingModel.__new__(HFEmbeddingModel)
    model.pooling, model.output_key = mode, key
    return model


def test_pooled_outputs_are_preserved_and_token_outputs_need_a_recipe():
    vectors = torch.tensor([[1., 2.]])
    torch.testing.assert_close(pooling()._pool({'pooler_output': vectors}, {}), vectors)
    tokens = torch.arange(24.).reshape(2, 4, 3)
    with pytest.raises(ValueError, match='choose pooling'):
        pooling()._pool({'last_hidden_state': tokens}, {})
    torch.testing.assert_close(pooling('cls')._pool({'last_hidden_state': tokens}, {}), tokens[:, 0])
    with pytest.raises(ValueError, match='output_key'):
        pooling()._pool({'logits': vectors}, {})
    torch.testing.assert_close(pooling(key='embedding')._pool({'embedding': vectors}, {}), vectors)


def test_mean_and_last_pooling_respect_left_and_right_padding():
    hidden = torch.arange(24.).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    output = SimpleNamespace(last_hidden_state=hidden)
    mean = pooling('mean')._pool(output, {'attention_mask': mask})
    torch.testing.assert_close(mean, torch.stack([hidden[0, :2].mean(0), hidden[1, 2:].mean(0)]))
    last = pooling('last')._pool(output, {'attention_mask': mask})
    torch.testing.assert_close(last, torch.stack([hidden[0, 1], hidden[1, 3]]))
    with pytest.raises(ValueError, match='valid tokens'):
        pooling('mean')._pool(output, {'attention_mask': torch.zeros_like(mask)})


def test_hf_id_registry_and_dinov3_preset_do_not_need_a_new_registry_entry(monkeypatch):
    calls = []
    def record(self, model_id, **kwargs):
        calls.append((model_id, kwargs))
    monkeypatch.setattr(HFEmbeddingModel, '__init__', record)
    get_model('hf:any-organisation/any-supported-model', input_type='text', pooling='last')
    assert calls[0][0] == 'any-organisation/any-supported-model'
    assert calls[0][1]['pooling'] == 'last'
    get_model('dinov3')
    assert calls[1][0] == 'facebook/dinov3-vits16-pretrain-lvd1689m'
    assert calls[1][1]['pooling'] == 'cls'
    get_model('qwen3-embedding-0.6b')
    assert calls[2][0] == 'Qwen/Qwen3-Embedding-0.6B'
    assert calls[2][1]['pooling'] == 'last'
    assert calls[2][1]['normalize'] is True
    assert calls[2][1]['processor_kwargs']['padding_side'] == 'left'
    with pytest.raises(ValueError, match='followed'):
        get_model('hf:')


def test_optional_dependency_error_explains_installation(monkeypatch):
    import builtins
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == 'transformers':
            raise ImportError('missing')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(ImportError, match=r'glyph-discovery\[hf\]'):
        HFEmbeddingModel('example')


@pytest.fixture
def tiny_dinov3(tmp_path):
    transformers = pytest.importorskip('transformers')
    if not hasattr(transformers, 'DINOv3ViTModel'):
        pytest.skip('DINOv3 needs transformers>=4.56')
    torch.manual_seed(17)
    config = transformers.DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, patch_size=16)
    transformers.DINOv3ViTModel(config).eval().save_pretrained(tmp_path)
    processor_class = getattr(transformers, 'DINOv3ViTImageProcessor', None) or transformers.DINOv3ViTImageProcessorFast
    processor_class(size={'height': 32, 'width': 32}).save_pretrained(tmp_path)
    return tmp_path


def test_real_dinov3_adapter_matches_direct_cls_embeddings(tiny_dinov3):
    encoder = get_model('dinov3', model_id=str(tiny_dinov3), device='cpu')
    images = [Image.new('RGB', (40, 50), color='red'), Image.new('L', (60, 40), color=100)]
    actual = encoder.embed_batch(images)
    with torch.inference_mode():
        batch = encoder.processor(images=[image.convert('RGB') for image in images], return_tensors='pt')
        expected = encoder.model(**batch).last_hidden_state[:, 0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 32) and encoder.get_embedding_dim() == 32
    torch.testing.assert_close(encoder.embed(images[0]), actual[0], rtol=1e-6, atol=1e-6)


def test_real_image_extraction_from_hf_id_keeps_ids_and_encoder_options(tiny_dinov3, tmp_path):
    from glyph.extraction import extract_embeddings
    image_path = tmp_path / 'image.png'
    Image.new('RGB', (40, 40), color='blue').save(image_path)
    metadata = pd.DataFrame(dict(sample_id=['second', 'first'], image_path=[str(image_path)] * 2))
    result = extract_embeddings(metadata, {'image_path': 'hf:' + str(tiny_dinov3)},
        model_kwargs={'image_path': {'input_type': 'image', 'pooling': 'cls'}}, batch_size=2)
    assert result['train']['sample_ids'] == ['second', 'first']
    assert result['train']['embeddings']['image_path'].shape == (2, 32)


def test_real_text_embeddings_use_tokenizer_and_masked_pooling(tmp_path):
    transformers = pytest.importorskip('transformers')
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n')
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(tmp_path)
    model = transformers.BertModel(transformers.BertConfig(vocab_size=7, hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)).eval()
    model.save_pretrained(tmp_path)
    encoder = get_model('hf', model_id=str(tmp_path), input_type='text', pooling='mean',
        read_text_files=False, normalize=True)
    actual = encoder.embed_batch(['hello', 'hello world'])
    with torch.inference_mode():
        batch = tokenizer(['hello', 'hello world'], padding=True, return_tensors='pt')
        hidden = model(**batch).last_hidden_state
        weights = batch.attention_mask.unsqueeze(-1)
        expected = (hidden * weights).sum(1) / weights.sum(1)
        expected = torch.nn.functional.normalize(expected, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(2))


def test_preprocessed_inputs_support_model_specific_embedding_methods(tiny_dinov3):
    encoder = get_model('hf', model_id=str(tiny_dinov3), input_type='processed', pooling='cls')
    result = encoder.embed_batch([{'pixel_values': torch.zeros(3, 32, 32)}])
    assert result.shape == (1, 32)


def test_user_owned_model_can_select_an_arbitrary_output_field():
    from glyph.models import EmbeddingAdapter
    class Custom(torch.nn.Module):
        def forward(self, values):
            return {'my_features': values * 2, 'prediction': values.sum(1)}
    encoder = EmbeddingAdapter(Custom(), output_key='my_features')
    actual = encoder.embed_batch([[1., 2.], [3., 4.]])
    torch.testing.assert_close(actual, torch.tensor([[2., 4.], [6., 8.]]))
    assert encoder.get_embedding_dim() == 2
    encoder.output_key = 'absent'
    with pytest.raises(ValueError, match='my_features'):
        encoder.embed_batch([[1., 2.]])


def test_custom_preprocessing_and_embedding_method_work_without_transformers():
    from glyph.models import EmbeddingAdapter
    class Custom(torch.nn.Module):
        def features(self, values):
            return (values + 1,)
    encoder = EmbeddingAdapter(Custom(), preprocess=lambda batch: {'values': torch.tensor(batch)},
        embedding_method='features', output_key=0)
    actual = encoder.embed_batch([[1., 2.]])
    torch.testing.assert_close(actual, torch.tensor([[2., 3.]]))


def test_custom_encoders_work_in_the_shared_extraction_api():
    from glyph.models import EmbeddingAdapter
    from glyph.extraction import extract_embeddings
    class Custom(torch.nn.Module):
        def forward(self, values):
            return {'embedding': values + 1}
    encoder = EmbeddingAdapter(Custom(), output_key='embedding')
    metadata = pd.DataFrame(dict(sample_id=['b', 'a'], inputs=[[1., 2.], [3., 4.]]))
    result = extract_embeddings(metadata, {'inputs': encoder}, batch_size=2)
    assert result['train']['sample_ids'] == ['b', 'a']
    torch.testing.assert_close(result['train']['embeddings']['inputs'], torch.tensor([[2., 3.], [4., 5.]]))


def test_custom_outputs_must_have_consistent_dimensions_and_finite_rows():
    from glyph.models import EmbeddingAdapter
    encoder = EmbeddingAdapter(lambda batch: batch)
    encoder.embed_batch([[1., 2.]])
    with pytest.raises(ValueError, match='dimension changed'):
        encoder.embed_batch([[1., 2., 3.]])
    with pytest.raises(ValueError, match='finite'):
        encoder.embed_batch([[float('nan'), 2.]])


def test_small_qwen_preset_matches_direct_last_token_recipe(tmp_path):
    transformers = pytest.importorskip('transformers')
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n')
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab), padding_side='left',
        model_input_names=['input_ids', 'attention_mask'])
    tokenizer.save_pretrained(tmp_path)
    model = transformers.Qwen3Model(transformers.Qwen3Config(vocab_size=7, hidden_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, intermediate_size=64, pad_token_id=0)).eval()
    model.save_pretrained(tmp_path)
    encoder = get_model('qwen3-embedding-0.6b', model_id=str(tmp_path), read_text_files=False)
    actual = encoder.embed_batch(['hello', 'hello world'])
    with torch.inference_mode():
        inputs = tokenizer(['hello', 'hello world'], padding=True, return_tensors='pt')
        hidden = model(**inputs).last_hidden_state
        expected = torch.nn.functional.normalize(hidden[:, -1], dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert encoder.processor.padding_side == 'left'


def test_readme_learning_workflow_with_local_dinov3_checkpoint(tiny_dinov3, tmp_path):
    import ast,re
    from pathlib import Path
    from glyph import Glyph
    root=Path(__file__).parents[1]
    text=(root/'README.md').read_text()
    example=re.search(r'```python\n(.*?)```',text,re.S)[1]
    call=next(node.value for node in ast.parse(example).body
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='Glyph')
    overrides=ast.literal_eval(next(keyword.value for keyword in call.keywords if keyword.arg=='config_overrides'))
    rows=[]
    for i in range(20):
        path=tmp_path/f'image_{i}.png'
        Image.new('RGB',(40,40),(i*11,200-i*7,i*5)).save(path)
        rows.append(dict(sample_id=f'new_{i}',image_path=str(path),pneumonia=i%2,
            split='train' if i<8 else 'val' if i<14 else 'test'))
    csv=tmp_path/'metadata.csv';pd.DataFrame(rows).to_csv(csv,index=False)
    overrides=[value.replace('/path/to/metadata.csv',str(csv)).replace(
        'hf:facebook/dinov3-vits16-pretrain-lvd1689m','hf:'+str(tiny_dinov3)) for value in overrides]
    overrides += [f'data.output_dir={tmp_path/"runs"}',f'data.cache_dir={tmp_path/"cache"}',
        'data.num_workers=0','device=cpu','sae.top_ks=[2]','sae.matryoshka=[8]',
        'sae.max_steps=3','sae.use_ghost_grads=false','phewas.min_activations=1','autointerp.enabled=false']
    obj=Glyph(config_path='config',config_overrides=overrides)
    obj.fit();obj.hypothesis_search()
    paths=obj.visualize(outcome='pneumonia',split='test')
    assert Path(paths[0]).is_file()
    assert obj.phewas_results is not None and len(obj.phewas_results)>0


@pytest.mark.skipif(os.environ.get('GLYPH_DINOV3_REAL') != '1',
    reason='Requires approved access to the official DINOv3 checkpoint')
def test_real_dinov3_readme_workflow_on_public_digits(tmp_path):
    from sklearn.datasets import load_digits
    from glyph import Glyph
    digits=load_digits()
    rows=[]
    for label in [0,1]:
        indices=np.flatnonzero(digits.target==label)[:12]
        for j,index in enumerate(indices):
            path=tmp_path/f'digit_{label}_{j}.png'
            pixels=(digits.images[index]*255/16).astype(np.uint8)
            Image.fromarray(pixels).convert('RGB').save(path)
            rows.append(dict(sample_id=f'digit_{label}_{j}',image_path=str(path),is_digit_one=label,
                split='train' if j<6 else 'val' if j<9 else 'test'))
    csv=tmp_path/'public_digits.csv';pd.DataFrame(rows).to_csv(csv,index=False)
    obj=Glyph('config',config_overrides=[f'data.metadata={csv}','data.input_cols=[image_path]',
        'data.outcome_cols=[is_digit_one]',
        'data.embedding_map={image_path: "hf:facebook/dinov3-vits16-pretrain-lvd1689m"}',
        'data.embedding_model_kwargs={image_path: {input_type: image, pooling: cls}}',
        f'data.output_dir={tmp_path/"runs"}',f'data.cache_dir={tmp_path/"cache"}',
        'data.num_workers=0','device=cpu','sae.top_ks=[2]','sae.matryoshka=[8]',
        'sae.max_steps=3','sae.use_ghost_grads=false','phewas.min_activations=1','autointerp.enabled=false'])
    obj.fit();obj.hypothesis_search()
    paths=obj.visualize(outcome='is_digit_one',split='test')
    assert Path(paths[0]).is_file()
