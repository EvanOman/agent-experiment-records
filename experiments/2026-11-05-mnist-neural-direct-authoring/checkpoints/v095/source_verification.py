"""Static source, inverse and raw-tie verification; no model execution."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'candidates/v094'


def verify(parent, graph, weights, provenance, info):
    assert len(parent['nodes']) == 113943 and len(graph['nodes']) == 125406
    assert len(weights) == 555 and sum(v.size for v in weights.values()) == 2471932
    assert sum(provenance['authored_specifications'].values()) == 1509
    inverse = deepcopy(graph)
    inverse['nodes'] = inverse['nodes'][:-11463]
    assert inverse['nodes'][-1]['name'] == 'on95_parent_logits'
    inverse['nodes'][-1]['name'] = 'logits'
    inverse['outputs'] = inverse['outputs'][:-112]
    inverse.pop('open_nine_readout')
    assert inverse == parent
    assert (json.dumps(inverse, indent=2) + '\n').encode() == (BASE/'graph.json').read_bytes()
    for name, value in weights.items():
        assert value.dtype == np.float32 and np.isfinite(value).all(), name
        expected = dict(dtype=str(value.dtype),shape=list(value.shape),entries=value.size,
                        raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
        assert provenance['all_tensor_records'][name] == expected, name
    assert len(info['feature_weights']) == 8 and sum(v.size for v in info['feature_weights'].values()) == 62
    for name, value in info['feature_weights'].items():
        inherited = weights[name]
        assert value.shape == inherited.shape and value.dtype == inherited.dtype
        assert value.tobytes() == inherited.tobytes(), name
    extra = graph['nodes'][113943:]
    feature = info['feature']
    feature_names = {n['name'] for n in feature['nodes']}
    for copy in info['copies']:
        actual = extra[copy['start']:copy['stop']]
        assert len(actual) == 1236
        reverse = {v:k for k,v in copy['boundary'].items()}
        for old, new in zip(feature['nodes'], actual):
            restored = deepcopy(new)
            assert restored['name'].startswith(copy['prefix'])
            restored['name'] = restored['name'][len(copy['prefix']):]
            restored['inputs'] = [reverse.get(k,k[len(copy['prefix']):] if k.startswith(copy['prefix']) else k) for k in restored['inputs']]
            assert restored == old, new['name']
            assert set(old['inputs']) <= feature_names | set(copy['boundary']) | set(info['feature_weights'])
    known = {'pixels', *weights}
    for node in graph['nodes']:
        assert node['name'] not in known and set(node['inputs']) <= known, node['name']
        known.add(node['name'])
    assert set(graph['outputs']) <= known and len(graph['outputs']) == len(set(graph['outputs'])) == 252
    old = {n['name']:n for n in parent['nodes']}
    new = {n['name']:n for n in graph['nodes']}
    assertions = 0

    def exact(bank, name, op, *inputs, **attrs):
        nonlocal assertions
        assert bank[name] == dict(name=name,op=op,inputs=list(inputs),**attrs), name
        assertions += 1

    rows = {121:[0,0,0,0,0,0,0,-3,0,5],187:[0,0,0,0,0,0,0,-3,0,3],154:[0,0,0,0,4,0,0,-2,0,0]}
    for row, values in rows.items():
        assert np.array_equal(weights['class_weights_v030'][row],np.array(values,np.float32))
    assert np.array_equal(weights['pr_class_weights'],np.array([[0,0,0,0,0,0,0,-3,0,0]],np.float32))
    exact(old,'sc80_body_row','sub','zero','pr_class_weights')
    for (view, branch), a in info['aliases'].items():
        p = a['prefix']
        stem = f'on95_{view}_{branch}_'
        exact(old,a['H'],'relu',p+'v21_evidence_difference')
        exact(old,p+'v21_evidence_difference','sub',a['R'],a['L'])
        exact(old,a['old_body_product'],'mul',a['T'],a['B'])
        exact(old,a['old_head_product'],'mul',a['T'],a['H'])
        exact(old,a['T'],'minimum',a['T'].replace('Tnew','cap'),p+'v25_not_bowl')
        exact(new,stem+'negative_clean_P','sub','zero',a['P'])
        exact(new,stem+'absent','step',stem+'negative_clean_P')
        exact(new,stem+'Qopen','minimum','on95_Qglobal',stem+'absent')
        exact(new,stem+'Qtail','minimum',stem+'Qopen',a['tail'])
        exact(new,stem+'short_permission','sub','one',a['short'])
        exact(new,stem+'Qshort','minimum',stem+'Qtail',stem+'short_permission')
        exact(new,stem+'two_permission','sub','one',a['two'])
        exact(new,stem+'Qeligible','minimum',stem+'Qshort',stem+'two_permission')
        exact(new,stem+'Bnew','maximum',a['B'],stem+'Qeligible')
        exact(new,stem+'Rnew','maximum',a['R'],stem+'Qeligible')
        exact(new,stem+'head_difference','sub',stem+'Rnew',a['L'])
        exact(new,stem+'Hnew','relu',stem+'head_difference')
        exact(new,stem+'Cnew','mul',a['C'],stem+'cup_permission')
        exact(new,stem+'body_restore_delta','sub',stem+'body_product_new',a['old_body_product'])
        exact(new,stem+'head_restore_delta','sub',stem+'head_product_new',a['old_head_product'])
        exact(new,stem+'restoration_delta','add',stem+'body_restore_delta',stem+'head_restore_delta')
    for view, vp in [('identity',''),('plus','re88_plus_'),('minus','re88_minus_')]:
        exact(old,vp+'oc65_r9_positive','maximum',vp+'oc65_r9','zero')
        exact(old,vp+'oc65_r9_permitted','mul',vp+'oc65_permission',vp+'oc65_r9_positive')
        exact(new,f'on95_{view}_newPerm','mul',f'on95_{view}_orientation_permission',vp+'oc65_r9_positive')
        exact(new,f'on95_{view}_orientDelta','sub',f'on95_{view}_newPerm',vp+'oc65_r9_permitted')
    exact(new,'on95_orientation_correction','pad','on95_orientation_column','zero',padding=[[0,0],[9,0]])
    exact(new,'on95_body_head_correction','add','on95_body_correction','on95_head_correction')
    exact(new,'on95_body_head_cup_correction','add','on95_body_head_correction','on95_cup_correction')
    exact(new,'on95_with_restoration_correction','add','on95_body_head_cup_correction','on95_restoration_correction')
    exact(new,'on95_correction','add','on95_with_restoration_correction','on95_orientation_correction')
    exact(new,'logits','add','on95_parent_logits','on95_correction')
    return dict(status='PASS',parent_inverse_serialized_bytes_exact=True,all113943_parent_nodes_preserved=True,
        all9_feature_inverses_exact=True,feature_nodes_inverse_checked=11124,all8_feature_arrays_byte_tied=True,
        all555_parent_array_records_equal=True,exact_named_source_checks=assertions,finite_topological_DAG=True,
        exact_rows=rows,orientation_structural_left_pad=True,all_parent_outputs_features_preserved=True,
        added_numerical_values=0,added_architecture='Fixed root-authored pooling, local eligibility and readout composition',
        note='Static construction checks only. Actual staged generic execution and independent peer review precede freeze.')
