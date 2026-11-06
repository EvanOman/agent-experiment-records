"""Source, algebra and generic execution checks only; no data or scores."""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import itertools
import json

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "candidates/v086"
BRANCHES = (("center", ""), ("low", "mt_low_"), ("high", "mt_high_"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bits(a, b, label=""):
    assert type(a) is type(b), (label, type(a), type(b))
    assert a.dtype == b.dtype and a.shape == b.shape, label
    assert a.tobytes(order="C") == b.tobytes(order="C"), label


def module_for(path, observed=False):
    if observed:
        source = (path / "runner.py").read_text()
        marker = '            values[node["name"]] = out\n'
        assert source.count(marker) == 1
        source = source.replace(marker, marker + '            yield node["name"], out\n')
        namespace = {"__name__": "v087_observed_runner", "__file__": str(path / "runner.py")}
        exec(compile(source, str(path / "runner.py") + "[observed]", "exec"), namespace)
        return namespace["Network"]
    spec = importlib.util.spec_from_file_location("v087_generic_" + path.name, path / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Network


def run(graph, weights, images, external=None, source=BASE, observed=False):
    cls = module_for(source, observed)
    network = cls.__new__(cls)
    network.graph, network.weights = graph, dict(weights)
    if external:
        assert not set(external) & set(weights)
        network.weights.update(external)
    return network.forward(images, diagnostics=True)


def small_graph(nodes, outputs=None):
    names = [n["name"] for n in nodes]
    return dict(nodes=deepcopy(nodes), outputs=names if outputs is None else list(outputs), prediction_logits=names[-1])


def cut(graph, roots, boundary):
    needed, kept = set(roots), []
    for n in reversed(graph["nodes"]):
        if n["name"] in needed and n["name"] not in boundary:
            kept.append(deepcopy(n))
            needed.update(n["inputs"])
    return kept[::-1]


def structural(parent,graph,weights,provenance,info):
    old_count=len(parent['nodes']);inverse=deepcopy(graph);inverse['nodes']=inverse['nodes'][:old_count]
    for n in inverse['nodes']:
        if n['name']=='co87_parent_logits':n['name']='logits'
    for key in ['outputs','diagnostic_scalars']:
        if key in parent:inverse[key]=deepcopy(parent[key])
        else:inverse.pop(key,None)
    assert inverse==parent and (json.dumps(inverse,indent=2)+'\n').encode()==(BASE/'graph.json').read_bytes()
    assert graph['feature_names']==parent['feature_names'] and len(graph['feature_names'])==302
    assert old_count==37422 and len(graph['nodes'])==37443
    assert graph['nodes'][old_count:-1]==info['body'] and len(info['body'])==20
    source=info['source']['graph'];source_nodes={n['name']:n for n in source['nodes']}
    for n in info['body']:assert n==source_nodes[n['name']],n['name']
    assert graph['nodes'][-1]==dict(name='logits',op='add',inputs=['co87_parent_logits','cp85_correction'])
    boundary={'pixels',*weights,*(n['name'] for n in graph['nodes'][:old_count])}
    assert cut(graph,['logits'],boundary)==graph['nodes'][old_count:]
    assert not any('logits' in key for n in info['body'] for key in [n['name'],*n['inputs']])
    assert sum('cp85_correction' in n['inputs'] for n in graph['nodes'])==1
    assert source_nodes['cp85_row']==dict(name='cp85_row',op='slice',inputs=['class_weights_v030'],slices=[[196,197],[None,None]])
    assert source_nodes['cp85_side']==dict(name='cp85_side',op='add',inputs=['cp85_low_weighted','cp85_high_weighted'])
    assert source_nodes['cp85_delta_mix']==dict(name='cp85_delta_mix',op='add',inputs=['cp85_center_weighted','cp85_side'])
    assert source_nodes['cp85_delta_column']==dict(name='cp85_delta_column',op='reshape',inputs=['cp85_delta_mix'],shape=[-1,1])
    assert source_nodes['cp85_correction']==dict(name='cp85_correction',op='affine',inputs=['cp85_delta_column','cp85_row','zero'])
    pnodes={n['name']:n for n in parent['nodes']};common=info['common']['graph']
    for n in common['nodes']:
        if n['name']!='logits':assert n==pnodes[n['name']]==source_nodes[n['name']]
    restored=deepcopy(source);restored['nodes']=restored['nodes'][:len(common['nodes'])];restored['outputs']=common['outputs']
    for n in restored['nodes']:
        if n['name']=='cp85_parent_logits':n['name']='logits'
    assert restored==common and (json.dumps(restored,indent=2)+'\n').encode()==(ROOT/'candidates/v079/graph.json').read_bytes()
    for b,p in BRANCHES:
        old=p+'at_qualified_closed_three';raw=p+'v19_closed_three';stem='cp85_'+b+'_'
        assert pnodes[old]==source_nodes[old]==dict(name=old,op='minimum',inputs=[raw,p+'at_three_permission'])
        head=pnodes['mt_'+b+'_logits'];assert head['inputs'][1]=='class_weights_v030'
        assert pnodes[head['inputs'][0]]['inputs'][196]==old
        expected=[dict(name=stem+'permission',op='sub',inputs=['one','os73_'+b+'_Qoutside']),
                  dict(name=stem+'permitted_raw',op='mul',inputs=[raw,stem+'permission']),
                  dict(name=stem+'new',op='maximum',inputs=[old,stem+'permitted_raw']),
                  dict(name=stem+'delta',op='sub',inputs=[stem+'new',old]),
                  dict(name=stem+'weighted',op='mul',inputs=[stem+'delta','mt_center_weight' if b=='center' else 'mt_side_weight'])]
        for n in expected:assert source_nodes[n['name']]==n
    assert len(weights)==552 and set(weights)==set(info['base']['weights'])==set(info['source']['weights'])
    for other in [info['base'],info['source'],info['common']]:
        for name,value in other['weights'].items():bits(weights[name],value,'shared array '+name)
    assert sum(v.size for v in weights.values())==1242619 and sum(provenance['authored_specifications'].values())==1506
    assert provenance['new_independent_numerical_specifications']==provenance['new_expanded_tensor_elements']==0
    assert provenance['new_array_count']==0
    np.testing.assert_array_equal(weights['class_weights_v030'][196],np.asarray([0,0,0,4,0,0,0,0,0,0],np.float32))
    return dict(parent_inverse_byte_exact=True,source85_inverse_byte_exact=True,parent_nodes_retained=old_count,source_closure_nodes=20,
                all_common_v079_nonfinal_nodes_exact=len(common['nodes'])-1,all_552_arrays_unchanged=True,
                no_new_values_arrays_or_entries=True,actual_row196_baselines_preserved=True,all_added_nodes_live=True)


def simultaneous(parent,graph,weights,info):
    count=8;external={};weighted={};tvalues={};delta_values={};body_values={}
    constants={
        'U':[1,.875,.75,.625,.5,1,.75,1],
        'D':[.875,1,.625,.75,1,0,.5,1],
        'cap':[.75,.625,1,.875,1,1,.625,1],
        'notbowl':[1,.75,.875,1,1,1,.875,1],
        'T0':[.25,.125,.375,.5,.5,0,.25,0],
        'body9':[.5,.625,.75,.875,.75,1,.5,0],
        'body21':[.75,.5,.875,.625,.25,1,.75,1],
    }
    for i,(label,prefix) in enumerate([('center',''),('low','mt_low_'),('high','mt_high_')]):
        v={k:np.asarray(a,np.float32).copy() for k,a in constants.items()}
        for a in v.values():a[:4]=np.roll(a[:4],i)
        if i:
            v['U'][7]=0;v['body9'][7]=1 if i==1 else 0;v['body21'][7]=0
        external.update({'hh70_'+label+'_Unew':v['U'],prefix+'ep_lower_down':v['D'],
                         prefix+'cap_excess':v['cap'],prefix+'v25_not_bowl':v['notbowl'],
                         prefix+'v25_seven_terminal':v['T0'],prefix+'v9_nine_body':v['body9'],
                         prefix+'v21_new_evidence':v['body21']})
        T=np.minimum(np.minimum(np.minimum(v['U'],v['D']),v['cap']),v['notbowl'])
        terms={'terminal':T-v['T0'],'body9':T*v['body9'],'body21':T*v['body21']}
        tvalues[label]=T;delta_values[label]=terms;body_values[label]=(v['body9'],v['body21'])
        for route,value in terms.items():
            assert np.all(value[:4]!=0),(label,route)
            weighted[label,route]=value*weights['mt_center_weight' if i==0 else 'mt_side_weight']
    mix={r:weighted['center',r]+(weighted['low',r]+weighted['high',r]) for r in ['terminal','body9','body21']}
    rowT=np.asarray([[0,0,0,0,0,0,0,2,0,0]],np.float32)
    rowB=np.asarray([[0,0,0,0,0,0,0,3,0,0]],np.float32)
    projections={r:mix[r][:,None]@(rowT if r=='terminal' else rowB)+weights['zero'] for r in mix}
    expected7=(projections['terminal']+projections['body9'])+projections['body21']
    assert all(np.all(value[:4,7]>0) for value in projections.values())
    # A complete parent input containing actual positive-nine row contributions
    # and nonzero corrections from all six existing v077 routes.
    rawbase=(np.arange(count*10,dtype=np.float32).reshape(count,10)-21)/13
    rawmix={}
    for j,route in enumerate(['body9','body21']):
        rawmix[route]=body_values['center'][j]*weights['mt_center_weight']+(body_values['low'][j]*weights['mt_side_weight']+body_values['high'][j]*weights['mt_side_weight'])
    positive9a=rawmix['body9'][:,None]@weights['class_weights_v030'][121:122]+weights['zero']
    positive9b=rawmix['body21'][:,None]@weights['class_weights_v030'][187:188]+weights['zero']
    assert np.all(positive9a[:4,9]>0) and np.all(positive9b[:4,9]>0)
    complete70=(rawbase+positive9a)+positive9b
    external['co77_parent_logits']=complete70
    rows={'B':66,'q2':38,'A':195,'W':218,'L':62,'U':61};old_corrections={}
    for j,(route,index) in enumerate(rows.items()):
        v=np.asarray([.125,.25,.375,.5,.625,.75,.875,.125],np.float32)
        if route in ['B','W','L','U']:v=-v
        v=np.roll(v,j)
        c=v[:,None]@weights['class_weights_v030'][index:index+1]+weights['zero']
        assert np.all(np.any(c!=0,axis=1));old_corrections[route]=c
        external['ob71_correction' if route=='B' else 'q68_correction' if route=='q2' else 'co77_'+route+'_correction']=c
    q=np.asarray([[1,1,1],[.5,.5,.5],[.25,.5,.75],[.75,.25,.5],[0,0,0],[1,0,0],[.5,.25,.75],[1,1,0]],np.float32)
    external.update(dict(zip(['dr_Q','mt_low_dr_Q','mt_high_dr_Q'],q.T)))
    # Literal simultaneous lower residual: distinct branches and active permissions.
    E=np.asarray([[1,.875,.75],[.75,1,.875],[.875,.75,1],[.625,.875,.75],
                  [0,1,1],[1,1,1],[.5,.75,1],[.75,.5,.25]],np.float32)
    M=np.asarray([[.125,.25,.375],[.25,.125,.375],[.125,.375,.25],[.25,.125,.375],
                  [0,0,0],[1,1,1],[.125,.25,.375],[0,0,0]],np.float32)
    F=np.asarray([[1,.875,.75],[.75,1,.875],[.875,.75,1],[1,.5,.75],
                  [1,1,1],[1,1,1],[.5,1,.75],[1,1,1]],np.float32)
    O=np.asarray([[.875,1,.75],[1,.75,.875],[.75,.875,1],[.5,1,.75],
                  [1,1,1],[1,1,1],[1,.5,.75],[1,1,1]],np.float32)
    X=np.asarray([[.75,.875,1],[.875,1,.75],[1,.75,.875],[.75,.5,1],
                  [1,1,1],[1,1,1],[.75,1,.5],[1,1,1]],np.float32)
    consensus=np.minimum(E[:,0],np.minimum(E[:,1],E[:,2]))
    lowerbase=((M*F)*O)*X
    lowerunion=np.maximum(M,consensus[:,None]);lowernew=((lowerunion*F)*O)*X
    lowerdelta=lowernew-lowerbase
    lowerweights=lowerdelta*np.asarray([.5,.25,.25],np.float32)
    lowermix=lowerweights[:,0]+(lowerweights[:,1]+lowerweights[:,2])
    lowercorr=lowermix[:,None]@np.asarray([[0,0,0,0,0,0,0,0,6,-3]],np.float32)+weights['zero']
    external.update(dict(zip(['fe_Q_extent','fel_fe_Q_extent','feh_fe_Q_extent'],E.T)))
    for i,(label,prefix) in enumerate(BRANCHES):
        external.update({f'co58_{label}_effective_lower':M[:,i],f'co58_{label}_permission':F[:,i],
                         f'co67_{label}_not_Q62':O[:,i],f'os73_{label}_Qoutside':1-X[:,i],
                         f'co77_{label}_Lnew':lowerbase[:,i]})
    assert np.all(lowercorr[:4,8]>0) and np.all(lowercorr[:4,9]<0)
    # The new source3 block uses the SAME outside signal as the parent lower gate.
    C_old=np.asarray([[.125,.25,.375],[.25,.125,.375],[.125,.375,.25],[.25,.125,.375],
                      [0,0,0],[1,1,1],[.5,.25,.375],[.75,.125,.25]],np.float32)
    C_raw=np.asarray([[1,.875,.75],[.875,.75,1],[.75,1,.875],[1,.875,.75],
                      [0,0,0],[1,1,1],[.75,.5,1],[1,.75,.5]],np.float32)
    C_permitted=C_raw*X;C_new=np.maximum(C_old,C_permitted);C_delta=C_new-C_old
    C_weighted=C_delta*np.asarray([.5,.25,.25],np.float32)
    C_mix=C_weighted[:,0]+(C_weighted[:,1]+C_weighted[:,2])
    C_correction=C_mix[:,None]@np.asarray([[0,0,0,4,0,0,0,0,0,0]],np.float32)+weights['zero']
    assert np.all(C_delta[:4]>0) and np.all(C_correction[:4,3]>0)
    for i,(label,prefix) in enumerate(BRANCHES):
        external.update({prefix+'at_qualified_closed_three':C_old[:,i],prefix+'v19_closed_three':C_raw[:,i]})
    parent_tail=cut(parent,['logits'],set(weights)|set(external))
    assert len(parent_tail)==88
    for n in parent_tail:
        if n['name']=='logits':n['name']='co87_parent_logits'
    result=run(small_graph(parent_tail+info['body']+graph['nodes'][-1:]),weights,np.zeros((count,28,28),np.uint8),external)
    oldcorr=(old_corrections['B']+old_corrections['q2'])+((old_corrections['A']+old_corrections['W'])+(old_corrections['L']+old_corrections['U']))
    complete77=complete70+oldcorr
    qmix=q[:,0]*weights['mt_center_weight']+(q[:,1]*weights['mt_side_weight']+q[:,2]*weights['mt_side_weight'])
    directed=qmix[:,None]@weights['dr74_class_weights']+weights['zero']
    complete79=(complete77+directed)+weights['to78_class_bias']
    assert np.all(directed[:4,3]!=0) and np.all((directed+weights['to78_class_bias'])[:4,3]!=0)
    complete84=complete79+lowercorr
    bits(result['fl84_parent_logits'],complete79,'complete79 calibrated parent retained')
    bits(result['fl84_Q_E'],consensus,'filled-lower minimum consensus')
    for i,(label,prefix) in enumerate(BRANCHES):
        for name,value in [('union',lowerunion),('after_flourish',lowerunion*F),
                           ('after_orphan',(lowerunion*F)*O),('Lnew',lowernew),('delta',lowerdelta)]:
            bits(result[f'fl84_{label}_{name}'],value[:,i],f'filled-lower {label} {name}')
    bits(result['fl84_mix'],lowermix,'filled-lower branch mixture')
    bits(result['fl84_correction'],lowercorr,'filled-lower projection')
    bits(result['co86_parent_logits'],complete84,'complete84 retained with all earlier routes')
    bits(result['sc80_correction'],expected7,'three-term source7 correction')
    complete86=complete84+expected7
    bits(result['co87_parent_logits'],complete86,'complete86 retained')
    for i,(label,prefix) in enumerate(BRANCHES):
        for name,value in [('permission',X),('permitted_raw',C_permitted),('new',C_new),('delta',C_delta),('weighted',C_weighted)]:
            bits(result[f'cp85_{label}_{name}'],value[:,i],f'new3 {label} {name}')
    bits(result['cp85_delta_mix'],C_mix,'new3 source mixture')
    bits(result['cp85_correction'],C_correction,'new3 source projection')
    bits(result['logits'],complete86+C_correction,'joint calibrated3 lower8/9 seven and new3')
    assert not np.array_equal(result['logits'],complete86)
    assert not np.array_equal(result['logits'],complete79+expected7)
    assert not np.array_equal(result['logits'],complete77+expected7)
    assert not np.array_equal(result['logits'],complete70+expected7)
    for label in ['center','low','high']:
        bits(result['sc80_'+label+'_Tnew'],tvalues[label],label+' nonlinear Tnew')
        for r in mix:
            key='sc80_'+label+'_'+('terminal_delta' if r=='terminal' else r+'_product')
            bits(result[key],delta_values[label][r],key)
    for r in mix:
        bits(result['sc80_'+r+'_mix'],mix[r],r+' branch mixture')
        bits(result['sc80_'+r+'_correction'],projections[r],r+' independent seven projection')
    unchanged=[i for i in range(10) if i!=3]
    np.testing.assert_array_equal(result['logits'][:,unchanged],complete86[:,unchanged])
    assert mix['terminal'][4]==0 and mix['body9'][4]>0 and mix['body21'][4]>0
    assert np.all(expected7[5]==0)
    assert tvalues['center'][6]==.5 and tvalues['center'][6]!=np.float32(.75*.5*.625*.875)
    assert mix['body9'][7]==0
    mixedT=tvalues['center'][7]*weights['mt_center_weight']+(tvalues['low'][7]*weights['mt_side_weight']+tvalues['high'][7]*weights['mt_side_weight'])
    assert mixedT*rawmix['body9'][7] != mix['body9'][7]
    return dict(records=count,simultaneous_nonzero_calibrated3_lower8_9_all_three_seven_and_new3=4,
                all_six_prior77_routes_active=True,positive_nine_and_other_columns_preserved=True,
                existing_terminal_with_body_cancellation=True,zero_terminal_positive_raw_bodies=True,
                fractional_min_and_product_before_mix=True,shared_outside_permissions_consistent=True,
                rejects_obsolete79_77_and70_heads=True,complete86_arithmetic_bitwise=True,
                new3_minimum_then_maximum_before_branch_mix=True)



def closed_three_fixtures(graph,weights,info):
    # Source-generic fixtures retained as literal expected values, not MNIST.
    old=np.asarray([[0,0,.25,0,.75,.5,1,.25,.25,.5,0,0],
                    [0,0,.25,0,.75,.5,0,.5,.5,.25,0,0],
                    [0,0,.25,0,.75,.5,0,.125,.75,0,0,0]],np.float32)
    raw=np.asarray([[0,1,1,1,1,0,1,1,1,.75,1,0],
                    [0,1,1,1,1,0,1,.75,1,1,0,0],
                    [0,1,1,1,1,0,0,1,1,.5,0,1]],np.float32)
    outside=np.asarray([[0,0,1,.5,.5,.25,1,.25,1,.5,0,1],
                        [0,0,1,.5,.5,.25,0,0,1,.5,1,1],
                        [0,0,1,.5,.5,.25,0,.5,1,0,1,0]],np.float32)
    delta=np.asarray([[0,1,0,.5,0,0,0,.5,0,0,1,0],
                      [0,1,0,.5,0,0,1,.25,0,.25,0,0],
                      [0,1,0,.5,0,0,0,.375,0,.5,0,1]],np.float32)
    mix=np.asarray([0,1,0,.5,0,0,.25,.40625,0,.1875,.5,.25],np.float32)
    base=np.arange(120,dtype=np.float32).reshape(12,10)/16-2
    external={'co87_parent_logits':base}
    for i,(b,p) in enumerate(BRANCHES):
        external.update({p+'at_qualified_closed_three':old[i],p+'v19_closed_three':raw[i],
                         'os73_'+b+'_Qoutside':outside[i]})
    result=run(small_graph(info['body']+graph['nodes'][-1:]),weights,np.zeros((12,28,28),np.uint8),external)
    for i,(b,p) in enumerate(BRANCHES):
        bits(result['cp85_'+b+'_delta'],delta[i],b+' literal delta')
        bits(result['cp85_'+b+'_new'],old[i]+delta[i],b+' retained floor')
    bits(result['cp85_delta_mix'],mix,'literal source branch mix')
    correction=np.zeros((12,10),np.float32);correction[:,3]=mix*4
    bits(result['cp85_correction'],correction,'pure class3 projection')
    bits(result['logits'],base+correction,'final complete-parent addition')
    np.testing.assert_array_equal(result['logits'][:,[0,1,2,4,5,6,7,8,9]],base[:,[0,1,2,4,5,6,7,8,9]])
    assert np.all(correction[:,3]>=0) and np.all(correction[:,3]<=4)
    for i in [0,2,4,5,8]:bits(result['logits'][i],base[i],'zero correction')
    # Mixing gates first gives .25, whereas max after mixing gives zero here.
    def mixed(v):return v[0]*np.float32(.5)+(v[1]*np.float32(.25)+v[2]*np.float32(.25))
    premature=np.maximum(mixed(old),mixed(raw*(1-outside)))-mixed(old)
    assert mix[6]==.25 and premature[6]==0
    # Preserve center+(low+high), distinguish the other float32 association.
    A=np.ones(2,np.float32);B=np.full(2,2**-24,np.float32);C=B.copy()
    baseline=np.zeros((2,10),np.float32);baseline[:,3]=-.5;baseline[:,7]=2;baseline[1,0]=-0.0
    external={'cp85_center_weighted':A,'cp85_low_weighted':B,'cp85_high_weighted':C,'co87_parent_logits':baseline}
    tail=cut(graph,['logits'],set(weights)|set(external))
    result=run(small_graph(tail),weights,np.zeros((2,28,28),np.uint8),external)
    assert (A+(B+C)).tobytes()!=((A+B)+C).tobytes()
    expected_mix=A+(B+C);expected=np.zeros((2,10),np.float32);expected[:,3]=expected_mix*4
    bits(result['cp85_delta_mix'],expected_mix,'source mix association')
    bits(result['cp85_correction'],expected,'source affine projection')
    bits(result['logits'],baseline+expected,'final addition and signed zero')
    assert np.signbit(baseline[1,0]) and not np.signbit(result['logits'][1,0])
    return dict(records=12,raw_zero_Q_zero_Q_one_fractional_old_dominates=True,old_floor_preserved=True,
                literal_expected_deltas_and_mixes=True,max_before_mixture=True,correction_bounds=[0,4],
                class3_only_delta=True,association_and_signed_zero_records=2)


def generic(parent,graph,weights,provenance,info):
    fields=np.zeros((4,28,28),np.uint8);fields[1]=255;fields[2,14,14]=255
    fields[3]=np.arange(784,dtype=np.uint16).reshape(28,28)%256
    names={n['name'] for n in info['body']}
    boundary={k for n in info['body'] for k in n['inputs'] if k not in names and k not in weights}
    captures,component_values={},{}
    left=run(parent,weights,fields,observed=True);right=run(graph,weights,fields,observed=True);count=0
    while True:
        try:name,a=next(left)
        except StopIteration as stopped:oldresult=stopped.value;break
        other,b=next(right);assert other==('co87_parent_logits' if name=='logits' else name)
        bits(a,b,'parent '+name);assert a.strides==b.strides,name
        if name in boundary:captures[name]=a
        count+=1
    while True:
        try:name,value=next(right)
        except StopIteration as stopped:result=stopped.value;break
        if name in names:component_values[name]=value
    assert count==37422 and set(captures)==boundary and set(component_values)==names
    source_result=run(small_graph(info['body']),info['source']['weights'],fields,captures,source=ROOT/'candidates/v085')
    for name,value in source_result.items():bits(component_values[name],value,'same-parent-fields source '+name)
    for name,value in oldresult.items():bits(result['co87_parent_logits' if name=='logits' else name],value,'parent output '+name)
    bits(result['logits'],oldresult['logits']+result['cp85_correction'],'actual generic final addition')
    ordinary=run(graph,weights,fields)
    for name,value in result.items():bits(ordinary[name],value,'uninstrumented '+name)
    assert np.isfinite(result['logits']).all()
    return dict(fields=['empty','full','impulse14_14','modulo_ramp'],
                parent_intermediate_tensors_bitwise_equal=count,source_intermediate_tensors_bitwise_equal=len(source_result),
                exact_source_boundary_names=sorted(boundary),same_actual_cleaned_parent_fields=True,
                all_parent_outputs_features_and_positive9_intermediates_preserved=True,uninstrumented_entrypoint_equal=True)


def verify(parent,graph,weights,provenance,info):
    report=dict(status='PASS',candidate='v087',parent='v086',source='v085',nodes=len(graph['nodes']),features=302,
                arrays=552,expanded=1242619,independent=1506,new_values_arrays_entries=0,
                author_sha256=sha(ROOT/'author_v087.py'),verifier_sha256=sha(Path(__file__)),
                structural=structural(parent,graph,weights,provenance,info),
                simultaneous=simultaneous(parent,graph,weights,info),
                closed_three=closed_three_fixtures(graph,weights,info),
                generic=generic(parent,graph,weights,provenance,info),
                no_MNIST_labels_results_archives_scoring_or_sealed_access=True)
    path=ROOT/'contributions/composition_v087_verification.json';assert not path.exists()
    path.write_text(json.dumps(report,indent=2)+'\n');return report


