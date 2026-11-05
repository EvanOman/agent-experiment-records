"""A fixed, pixel-only symmetric chamfer RBF network.

The class-specific numeric tensors are authored glyph masks and their distance
channels. All preprocessing and hidden operations are shared across classes.
No labels, data paths, fitting, parameter writing, or sample identity are used.
"""
import numpy as np
from scipy import ndimage
import shared_features
import part_metric
import part_metric_v2
import part_metric_v2_fast
import topology_features

VERSION = 'topology-multiview-parts-v15'

def normalize_one(raw, parameters):
    size = int(parameters['size'])
    a = np.asarray(raw, dtype=np.float32) / np.float32(255.0)
    if parameters.get('prethin_iterations',0):
        a=thin((a>parameters['ink_threshold'])[None],parameters['prethin_iterations'])[0]
    source=a
    yy, xx = np.indices(a.shape, dtype=np.float32)
    mass = a.sum() + np.float32(1e-8)
    xm, ym = (xx*a).sum()/mass, (yy*a).sum()/mass
    cov = ((xx-xm)*(yy-ym)*a).sum()
    var = ((yy-ym)**2*a).sum() + np.float32(1e-8)
    shear = float(np.clip(cov/var, -parameters['max_shear'], parameters['max_shear'])) * parameters['deskew']
    a = ndimage.affine_transform(a, [[1.,0.],[shear,1.]], offset=[0.,-shear*ym], order=1, mode='constant')
    mask = a > parameters['bbox_threshold']
    rows = np.flatnonzero(mask.any(axis=1)); cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows) or not len(cols): return np.zeros((size,size),np.float32)
    top,bottom,left,right = rows[0],rows[-1]+1,cols[0],cols[-1]+1
    height,width = bottom-top,right-left
    target_h = parameters['target_height']
    target_w = min(parameters['target_width_max'], target_h*(width/height)**parameters['aspect_power']*parameters['aspect_reference']**(1.-parameters['aspect_power']))
    sy,sx = height/target_h,width/target_w
    # Center bounding box and preserve floating point scaling, no sample-specific learned transform.
    offset=[(top+bottom-1)*.5-sy*(size-1)*.5,(left+right-1)*.5-sx*(size-1)*.5]
    if parameters.get('single_resample',False):
        combined_matrix=[[sy,0.],[shear*sy,sx]]
        combined_offset=[offset[0],shear*offset[0]+offset[1]-shear*ym]
        return ndimage.affine_transform(source,combined_matrix,offset=combined_offset,output_shape=(size,size),order=1,mode='constant').astype(np.float32)
    return ndimage.affine_transform(a, [[sy,0.],[0.,sx]],offset=offset,output_shape=(size,size),order=1,mode='constant').astype(np.float32)

def normalize(images, parameters):
    a=np.asarray(images)
    if a.ndim!=3 or a.shape[1:]!=(28,28) or a.dtype!=np.uint8:
        raise ValueError('Expected N x 28 x 28 uint8 pixels')
    return np.stack([normalize_one(x,parameters) for x in a])

def thin(ink, iterations):
    """Fixed unrolled shared two-stage morphological thinning graph."""
    x=ink.astype(bool).copy()
    for iteration in range(iterations):
        for phase in range(2):
            z=np.pad(x,((0,0),(1,1),(1,1)))
            n=[z[:, :-2,1:-1], z[:,:-2,2:], z[:,1:-1,2:], z[:,2:,2:], z[:,2:,1:-1], z[:,2:,:-2], z[:,1:-1,:-2], z[:,:-2,:-2]]
            count=sum(v.astype(np.int8) for v in n)
            changes=sum((~n[k]&n[(k+1)%8]).astype(np.int8) for k in range(8))
            if phase==0:
                preserve=(n[0]&n[2]&n[4]) | (n[2]&n[4]&n[6])
            else:
                preserve=(n[0]&n[2]&n[6]) | (n[0]&n[4]&n[6])
            remove=x & (count>=2) & (count<=6) & (changes==1) & ~preserve
            x=x & ~remove
    return x.astype(np.float32)

def distance_features(ink, power, cap):
    # Euclidean min-pooling: min over foreground sites of fixed squared distances.
    # The scipy implementation computes the same fixed spatial min-plus graph.
    return np.stack([np.minimum(ndimage.distance_transform_edt(~x.astype(bool))**power,cap) for x in ink]).astype(np.float32)

def enclosure(a, kernel):
    """Uniform four-direction max-pooling enclosure proxy, no class logic."""
    left=np.maximum.accumulate(a,axis=2)
    right=np.maximum.accumulate(a[:,:,::-1],axis=2)[:,:,::-1]
    up=np.maximum.accumulate(a,axis=1)
    down=np.maximum.accumulate(a[:,::-1,:],axis=1)[:,::-1,:]
    void=np.clip(np.minimum(np.minimum(left,right),np.minimum(up,down))-a,0.,1.)
    k=np.asarray(kernel,dtype=np.float32)
    return ndimage.correlate1d(ndimage.correlate1d(void,k,axis=1,mode='constant'),k,axis=2,mode='constant')

def features(images, weights):
    p=weights['parameters']
    a=normalize(images,p)
    ink=thin((a>p['ink_threshold']).astype(np.float32),p.get('thin_iterations',0))
    dist=distance_features(ink,p['distance_power'],p['distance_cap'])
    return {'normalized':a,'ink':ink,'distance':dist}

def part_features(a,p):
    metric=part_metric_v2_fast if p.get('part_fast',False) else (part_metric_v2 if p.get('part_metric_version',1)==2 else part_metric)
    feature=metric.extract(a,p['part_feature_parameters'])
    if p.get('part_energy_target',0):
        channel_weights=np.array(p['part_feature_parameters']['channel_weights'],dtype=np.float32)
        energy=(feature*feature*channel_weights[None,:,None,None]).sum(axis=1).mean(axis=(1,2))
        feature=feature*np.sqrt(p['part_energy_target']/np.maximum(energy,p['part_energy_epsilon']))[:,None,None,None]
    return feature

def _forward_one(images,weights,return_activations=False):
    p=weights['parameters']; f=features(images,weights)
    ink=f['ink'].reshape(len(images),-1); dist=f['distance'].reshape(len(images),-1)
    t=weights['template_ink']; td=weights['template_distance']
    a=(ink @ td.T)/np.maximum(ink.sum(axis=1,keepdims=True),1.)
    b=(dist @ t.T)/np.maximum(t.sum(axis=1)[None,:],1.)
    costs=p['input_coverage_weight']*a+p['template_coverage_weight']*b
    if p.get('gray_weight',0):
        kernel=np.array(p['gray_kernel'],dtype=np.float32)
        gray=ndimage.correlate1d(ndimage.correlate1d(f['normalized'],kernel,axis=1,mode='constant'),kernel,axis=2,mode='constant').reshape(len(images),-1)
        tg=weights['template_gray']
        energy=(gray*gray).sum(axis=1)[:,None]+(tg*tg).sum(axis=1)[None,:]
        gray_cost=(energy-2.*(gray@tg.T))/np.maximum(energy,1e-6)
        costs=costs+p['gray_weight']*gray_cost
    if p.get('enclosure_weight',0):
        h=enclosure(f['normalized'],p['enclosure_kernel']).reshape(len(images),-1)
        th=weights['template_enclosure']
        energy=(h*h).sum(axis=1)[:,None]+(th*th).sum(axis=1)[None,:]
        hole_cost=(energy-2.*(h@th.T))/np.maximum(energy,p['enclosure_epsilon'])
        costs=costs+p['enclosure_weight']*hole_cost
    if 'shared_feature_parameters' in p:
        feature_cost=shared_features.compare(shared_features.extract(f['normalized'],p['shared_feature_parameters']),weights['template_shared'],p['shared_feature_parameters'])
        costs=p['chamfer_weight']*costs+p['shared_weight']*feature_cost
    ensemble_scores=None
    if 'part_feature_parameters' in p:
        metric=part_metric_v2_fast if p.get('part_fast',False) else (part_metric_v2 if p.get('part_metric_version',1)==2 else part_metric)
        part_cost=metric.compare(part_features(f['normalized'],p),weights['template_parts'],p['part_feature_parameters'])
        if p.get('class_ensemble',False):
            geometric_scores=np.stack([-costs[:,weights['labels']==k].min(axis=1) for k in range(10)],axis=1)
            part_scores=np.stack([-part_cost[:,weights['labels']==k].min(axis=1) for k in range(10)],axis=1)
            ensemble_scores=p['chamfer_weight']*geometric_scores+p['part_weight']*part_scores
        costs=p['chamfer_weight']*costs+p['part_weight']*part_cost
    if 'topology_parameters' in p:
        topology=topology_features.extract(f['normalized'],p['topology_parameters'])
        costs=costs+topology_features.compare(topology,weights['template_topology'],p['topology_parameters'])
    scores=np.stack([-costs[:,weights['labels']==k].min(axis=1) for k in range(10)],axis=1)
    if ensemble_scores is not None:
        scores=ensemble_scores
    if 'class_score_bias' in p:
        scores=scores+np.asarray(p['class_score_bias'],dtype=np.float32)[None,:]
    if return_activations:
        return scores,{**f,'template_costs':costs,'winning_template':costs.argmin(axis=1)}
    return scores

def forward(images,weights,return_activations=False):
    p=weights['parameters']
    if 'deskew_views' not in p:
        return _forward_one(images,weights,return_activations)
    views=p['deskew_views']; coefficients=p['view_weights']
    if len(views)!=len(coefficients) or not np.isclose(sum(coefficients),1.):
        raise ValueError('Fixed view coefficients must match views and sum to one')
    scores=np.zeros((len(images),10),dtype=np.float32)
    view_scores=[]; representative=None
    for index,(deskew,coefficient) in enumerate(zip(views,coefficients)):
        local=dict(weights)
        local['parameters']={key:value for key,value in p.items() if key not in ('deskew_views','view_weights','class_score_bias')}
        local['parameters']['deskew']=deskew
        if index:
            prefix=f'view{index}_'
            local.update({key[len(prefix):]:value for key,value in weights.items() if key.startswith(prefix)})
        if return_activations and index==0:
            branch,representative=_forward_one(images,local,True)
        else:
            branch=_forward_one(images,local)
        view_scores.append(branch)
        scores+=np.float32(coefficient)*branch
    if 'class_score_bias' in p:
        scores+=np.asarray(p['class_score_bias'],dtype=np.float32)[None,:]
    if return_activations:
        return scores,{**representative,'view_scores':np.stack(view_scores),'representative_view':0}
    return scores

def predict(images,weights):
    return forward(images,weights).argmax(axis=1)

def load_weights(path):
    import json
    with np.load(path,allow_pickle=False) as z:
        out={k:z[k] for k in z.files if k!='parameters_json'}
        out['parameters']=json.loads(str(z['parameters_json']))
    return out
