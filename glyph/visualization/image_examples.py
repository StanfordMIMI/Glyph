"""Compact, shared thumbnails for activation-selected concept examples."""
import base64
from io import BytesIO
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def collect_image_examples(features, sample_ids, metadata, input_column, *,
                           feature_indices=None, split='train', count=2,
                           thumbnail_size=256, quality=75):
    """Select the highest positive activations, without using outcome labels.

    Each readable 2D image is encoded once and referenced by all its concepts.
    Non-image modalities and unavailable paths simply have no image examples.
    """
    if not isinstance(count,int) or isinstance(count,bool) or not 0<=count<=8:
        raise ValueError('image_examples must be an integer between zero and eight')
    if not isinstance(thumbnail_size,int) or not 32<=thumbnail_size<=1024:
        raise ValueError('thumbnail_size must be between 32 and 1024 pixels')
    bank={};examples={}
    result=dict(image_bank=bank,concept_images=examples,
                image_example_info=dict(split=str(split),count=count,selection='Highest positive activations; outcome labels are not used.'))
    if not count or metadata is None or input_column not in metadata:
        return result
    values=features.detach().cpu().numpy() if hasattr(features,'detach') else np.asarray(features)
    ids=np.asarray(sample_ids).astype(str)
    if values.ndim!=2 or len(values)!=len(ids):
        raise ValueError('Image example activations and sample IDs must align')
    frame=pd.DataFrame(metadata).copy()
    if 'sample_id' not in frame or frame.sample_id.isna().any() or frame.sample_id.astype(str).duplicated().any():
        raise ValueError('Image examples require unique nonmissing sample_id values')
    frame['sample_id']=frame.sample_id.astype(str);frame=frame.set_index('sample_id')
    indices=np.arange(values.shape[1]) if feature_indices is None else np.asarray(feature_indices,dtype=int)
    if len(indices) and (indices.min()<0 or indices.max()>=values.shape[1]):
        raise ValueError('Image example feature index is outside the dictionary')
    def is_candidate(sample):
        if sample not in frame.index:return False
        path=frame.loc[sample,input_column]
        if not isinstance(path,(str,Path)):return False
        try:return Path(path).suffix.lower() not in {'.nii','.gz','.json','.txt','.csv','.npz','.pt'} and Path(path).is_file()
        except (OSError,ValueError):return False
    positions=np.array([i for i,sample in enumerate(ids) if is_candidate(sample)],dtype=int)
    if not len(positions):return result
    scores=values[positions][:,indices]
    ordered=np.argsort(-np.where(np.isfinite(scores),scores,-np.inf),axis=0,kind='stable')
    cached={}
    def thumbnail(position):
        if position in cached:return cached[position]
        cached[position]=None;sample=ids[position]
        if sample not in frame.index:return None
        path=frame.loc[sample,input_column]
        if not isinstance(path,(str,Path)):return None
        try:
            with Image.open(path) as source:
                image=source.convert('RGB');pixelated=max(image.size)<=64
                image.thumbnail((thumbnail_size,thumbnail_size),Image.Resampling.LANCZOS)
                buffer=BytesIO();mime='png' if pixelated else 'jpeg'
                image.save(buffer,format='PNG' if pixelated else 'JPEG',**({} if pixelated else dict(quality=quality,optimize=True)))
                raw=buffer.getvalue();key=hashlib.sha256((sample+'\0'+str(path)).encode()).hexdigest()[:20]
                bank[key]=dict(src=f'data:image/{mime};base64,'+base64.b64encode(raw).decode('ascii'),
                    sample_id=sample,split=str(split),width=image.width,height=image.height,pixelated=pixelated)
                cached[position]=key
        except (OSError,ValueError,TypeError,Image.DecompressionBombError):
            return None
        return key
    for column,index in enumerate(indices):
        selected=[]
        for local_position in ordered[:,column]:
            position=int(positions[local_position])
            activation=float(values[position,index])
            if not np.isfinite(activation) or activation<=1e-8:break
            key=thumbnail(int(position))
            if key is not None:selected.append([key,activation])
            if len(selected)==count:break
        if selected:examples[str(index)]=selected
    return result


def image_payload(examples=None):
    """Validate embedded image references before writing a standalone viewer."""
    if not examples:return dict(image_bank={},concept_images={},image_example_info={})
    bank=examples.get('image_bank',{});concepts=examples.get('concept_images',{})
    for record in bank.values():
        source=record.get('src','')
        if not source.startswith(('data:image/png;base64,','data:image/jpeg;base64,')):
            raise ValueError('Explorer examples require embedded PNG or JPEG thumbnails')
        try:base64.b64decode(source.split(',',1)[1],validate=True)
        except (ValueError,TypeError) as exc:raise ValueError('Invalid embedded thumbnail') from exc
    for refs in concepts.values():
        for key,activation in refs:
            if key not in bank or not np.isfinite(activation):
                raise ValueError('Invalid concept image reference or activation')
    return dict(image_bank=bank,concept_images=concepts,image_example_info=examples.get('image_example_info',{}))
