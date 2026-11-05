"""Portable data-free expansion of the frozen agent-authored stroke parameters.

The two numerical functions below are copied byte-for-byte from the original
materializer. Only its original repository-path setup and authoring CLI were
removed. They do not read image datasets or fit model parameters.
"""
import numpy as np
from scipy import ndimage
import shared_features
import part_metric
import topology_features
from model import normalize_one, distance_features, thin, enclosure, part_features, VERSION

def rasterize(glyph):
    # Fixed supersampling canvas. Every stroke is a union of linear segments.
    yy,xx=np.indices((64,64),dtype=np.float32)
    points=np.stack([xx/63.,yy/63.],axis=-1)
    distance=np.full((64,64),np.inf,dtype=np.float32)
    for stroke in glyph['strokes']:
        for av,bv in zip(stroke[:-1],stroke[1:]):
            a,b=np.array(av),np.array(bv); v=b-a
            t=np.clip(((points-a)*v).sum(axis=-1)/max(float(v@v),1e-12),0.,1.)
            distance=np.minimum(distance,np.linalg.norm(points-a-t[...,None]*v,axis=-1))
    # One-pixel antialias boundary; literal width is the full stroke width.
    return (np.clip((glyph['width']*.5-distance)*63.+.5,0.,1.)*255.).astype(np.float32)

def template_fields(glyphs,parameters):
    normalized=np.stack([normalize_one(rasterize(g),parameters) for g in glyphs])
    ink=thin((normalized>parameters['ink_threshold']).astype(np.float32),parameters.get('thin_iterations',0))
    dist=distance_features(ink,parameters['distance_power'],parameters['distance_cap'])
    extra={'template_ink':ink.reshape(len(ink),-1),'template_distance':dist.reshape(len(dist),-1),'template_normalized':normalized}
    if 'topology_parameters' in parameters:
        extra['template_topology']=topology_features.extract(normalized,parameters['topology_parameters'])
    if 'part_feature_parameters' in parameters:
        extra['template_parts']=part_features(normalized,parameters)
    if parameters.get('enclosure_weight',0):
        extra['template_enclosure']=enclosure(normalized,parameters['enclosure_kernel']).reshape(len(normalized),-1)
    if parameters.get('gray_weight',0):
        kernel=np.array(parameters['gray_kernel'],dtype=np.float32)
        extra['template_gray']=ndimage.correlate1d(ndimage.correlate1d(normalized,kernel,axis=1,mode='constant'),kernel,axis=2,mode='constant').reshape(len(normalized),-1)
    if 'shared_feature_parameters' in parameters:
        extra['template_shared']=shared_features.extract(normalized,parameters['shared_feature_parameters'])
    return extra

