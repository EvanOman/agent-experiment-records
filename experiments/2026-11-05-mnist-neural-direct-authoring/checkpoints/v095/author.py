"""Append the one root-authored open-head readout to complete frozen v094."""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
if ROOT.name == 'v095' and ROOT.parent.name == 'candidates':
    ROOT = ROOT.parent.parent
BASE = ROOT / 'candidates/v094'
DEST = ROOT / 'candidates/v095'
FEATURE = ROOT / 'contributions/open_nine_path_feature_frozen'
STAGE = ROOT / 'runtime/open_nine_head_v095_staging'
PREFIX = ROOT / 'contributions/open_nine_head_v095'
SPEC_SHA = 'a1dfe30afcdce6eb7529f8ebb2ead5aa3e8f931e56e8f73b092882258709dbc2'
PARENT_MANIFEST_SHA = '35eb677ebdf71327b05dd11da1067f8ee144ad44173f7b7658ede176d45fdf42'
FEATURE_MANIFEST_SHA = '9a85226d411fa1087978ae3bb6fad3c067819407dd43482ecf35d69e8688fa49'
SOURCE_SHA = '7b5e21290b595573f5f0e2e7e9ec9afbdd2620401a77027b13285e6ba9223de6'
SOURCE_PEER_SHA = '3ab07c5d6b3cd6ca7cc1e6d3cc2f6ae0870db8f03030a637e59255b410593fb6'
VIEWS = [('identity', ''), ('plus', 're88_plus_'), ('minus', 're88_minus_')]
THRESHOLDS = [('center', ''), ('low', 'mt_low_'), ('high', 'mt_high_')]
FAMILIES = ['body', 'head', 'cup', 'restoration']
ROWS = {'body': 121, 'head': 187, 'cup': 154}
FEATURE_ARRAYS = ['zero', 'one', 'ep_half', 'quarter', 'hole_gain', 'row_indices', 'ep_column_coordinates', 'ep_image_width']
FEATURE_INPUTS = dict(S='ep_clean_skeleton', E='ep_endpoints', Mx='v17_moment_x', My='v17_moment_y', U='v17_cup_row_map', D='ep_lower_down_map', h='ov_extent_denominator')
PARENT_NODES = 113943
GEOMETRY_NODES = 11124
ADDED_NODES = 11463
TOTAL_NODES = 125406
EXTRA_OUTPUTS = ['on95_parent_logits', 'on95_Qglobal'] + [f'on95_{v}_median' for v, _ in VIEWS]
EXTRA_OUTPUTS += [f'on95_{v}_{b}_geom_Q' for v, _ in VIEWS for b, _ in THRESHOLDS]
EXTRA_OUTPUTS += [f'on95_{v}_{b}_{s}' for v, _ in VIEWS for b, _ in THRESHOLDS for s in ['Qeligible', 'Bnew', 'Rnew', 'Hnew', 'Cnew', 'body_delta', 'head_delta', 'cup_delta', 'restoration_delta']]
EXTRA_OUTPUTS += [f'on95_{v}_{s}' for v, _ in VIEWS for s in ['CmixNew', 'orientDelta']]
EXTRA_OUTPUTS += [f'on95_{f}_{s}' for f in FAMILIES for s in ['mix', 'correction']]
EXTRA_OUTPUTS += ['on95_orientation_mix', 'on95_orientation_correction', 'on95_correction']
assert len(EXTRA_OUTPUTS) == 112


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(value):
    return dict(dtype=str(value.dtype), shape=list(value.shape), entries=value.size,
                raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def load(path, expected_manifest):
    assert sha(path / 'SHA256SUMS.json') == expected_manifest
    manifest = json.loads((path / 'SHA256SUMS.json').read_text())
    for name, digest in manifest.items():
        assert sha(path / name) == digest, name
    graph = json.loads((path / 'graph.json').read_text())
    with np.load(path / 'weights.npz', allow_pickle=False) as archive:
        weights = {k: archive[k] for k in archive.files}
    return graph, weights, manifest


def load_aliases(parent):
    path = ROOT / 'contributions/open_nine_readout_source_audit.json'
    peer = ROOT / 'contributions/open_nine_readout_source_peer_audit.json'
    assert sha(path) == SOURCE_SHA and sha(peer) == SOURCE_PEER_SHA
    source = json.loads(path.read_text())
    assert source['candidate_manifest_sha256'] == PARENT_MANIFEST_SHA
    nodes = {n['name']: n for n in parent['nodes']}
    aliases = {}
    for view, vp in VIEWS:
        for branch, bp in THRESHOLDS:
            a = source['all_nine_branch_bindings'][view + '_' + branch]
            assert a['ordinary_prefix'] == vp + bp
            ordinary = a['aliases']
            aliases[(view, branch)] = dict(
                prefix=vp + bp, B=ordinary['v9_nine_body'], R=ordinary['v21_route'],
                H=ordinary['v21_new_evidence'], L=ordinary['ep_nine_loop_terminal'],
                C=ordinary['v17_open_cup_four'], P=vp + bp + 'ov_clean_P',
                tail=ordinary['ep_lower_down'], short=vp + bp + 'ov_shortpath',
                two=ordinary['supported_two_holes'], T=a['sc80']['Tnew'],
                old_body_product=a['sc80']['body9_product'],
                old_head_product=a['sc80']['body21_product'],
                feature_boundary={key: vp + bp + value for key, value in FEATURE_INPUTS.items()},
            )
            for key, value in aliases[(view, branch)].items():
                if key not in ['prefix', 'feature_boundary']:
                    assert value in nodes, value
            assert set(aliases[(view, branch)]['feature_boundary'].values()) <= set(nodes)
    return aliases


def suffix(feature, aliases):
    nodes, copies = [], []
    feature_names = {n['name'] for n in feature['nodes']}

    def add(name, op, *inputs, **attrs):
        nodes.append(dict(name=name, op=op, inputs=list(inputs), **attrs))

    for view, _ in VIEWS:
        for branch, _ in THRESHOLDS:
            stem = f'on95_{view}_{branch}_geom_'
            boundary = aliases[(view, branch)]['feature_boundary']
            start = len(nodes)
            for source in feature['nodes']:
                item = deepcopy(source)
                item['name'] = stem + source['name']
                item['inputs'] = [boundary.get(k, stem + k if k in feature_names else k) for k in source['inputs']]
                nodes.append(item)
            copies.append(dict(view=view, threshold=branch, prefix=stem, start=start, stop=len(nodes), boundary=boundary))
    assert len(nodes) == GEOMETRY_NODES
    for view, _ in VIEWS:
        stem = f'on95_{view}_'
        a, b, c = [f'on95_{view}_{branch}_geom_Q' for branch, _ in THRESHOLDS]
        add(stem + 'ab_min', 'minimum', a, b)
        add(stem + 'ab_max', 'maximum', a, b)
        add(stem + 'upper_pair_min', 'minimum', stem + 'ab_max', c)
        add(stem + 'median', 'maximum', stem + 'ab_min', stem + 'upper_pair_min')
    add('on95_side_max', 'maximum', 'on95_plus_median', 'on95_minus_median')
    add('on95_Qglobal', 'maximum', 'on95_identity_median', 'on95_side_max')
    for view, _ in VIEWS:
        for branch, _ in THRESHOLDS:
            stem = f'on95_{view}_{branch}_'
            a = aliases[(view, branch)]
            add(stem + 'negative_clean_P', 'sub', 'zero', a['P'])
            add(stem + 'absent', 'step', stem + 'negative_clean_P')
            add(stem + 'Qopen', 'minimum', 'on95_Qglobal', stem + 'absent')
            add(stem + 'Qtail', 'minimum', stem + 'Qopen', a['tail'])
            add(stem + 'short_permission', 'sub', 'one', a['short'])
            add(stem + 'Qshort', 'minimum', stem + 'Qtail', stem + 'short_permission')
            add(stem + 'two_permission', 'sub', 'one', a['two'])
            add(stem + 'Qeligible', 'minimum', stem + 'Qshort', stem + 'two_permission')
            add(stem + 'Bnew', 'maximum', a['B'], stem + 'Qeligible')
            add(stem + 'Rnew', 'maximum', a['R'], stem + 'Qeligible')
            add(stem + 'head_difference', 'sub', stem + 'Rnew', a['L'])
            add(stem + 'Hnew', 'relu', stem + 'head_difference')
            add(stem + 'cup_permission', 'sub', 'one', stem + 'Qeligible')
            add(stem + 'Cnew', 'mul', a['C'], stem + 'cup_permission')
            for family, new, old in [('body', 'Bnew', 'B'), ('head', 'Hnew', 'H'), ('cup', 'Cnew', 'C')]:
                add(stem + family + '_delta', 'sub', stem + new, a[old])
            add(stem + 'body_product_new', 'mul', a['T'], stem + 'Bnew')
            add(stem + 'body_restore_delta', 'sub', stem + 'body_product_new', a['old_body_product'])
            add(stem + 'head_product_new', 'mul', a['T'], stem + 'Hnew')
            add(stem + 'head_restore_delta', 'sub', stem + 'head_product_new', a['old_head_product'])
            add(stem + 'restoration_delta', 'add', stem + 'body_restore_delta', stem + 'head_restore_delta')
            weight = 'mt_center_weight' if branch == 'center' else 'mt_side_weight'
            for family in FAMILIES:
                add(stem + family + '_weighted', 'mul', stem + family + '_delta', weight)
            add(stem + 'Cnew_weighted', 'mul', stem + 'Cnew', weight)
    for view, vp in VIEWS:
        stem = f'on95_{view}_'
        for family in FAMILIES:
            add(stem + family + '_sides', 'add', stem + 'low_' + family + '_weighted', stem + 'high_' + family + '_weighted')
            add(stem + family + '_mix', 'add', stem + 'center_' + family + '_weighted', stem + family + '_sides')
        add(stem + 'Cnew_sides', 'add', stem + 'low_Cnew_weighted', stem + 'high_Cnew_weighted')
        add(stem + 'CmixNew', 'add', stem + 'center_Cnew_weighted', stem + 'Cnew_sides')
        add(stem + 'orientation_permission', 'sub', 'one', stem + 'CmixNew')
        add(stem + 'newPerm', 'mul', stem + 'orientation_permission', vp + 'oc65_r9_positive')
        add(stem + 'orientDelta', 'sub', stem + 'newPerm', vp + 'oc65_r9_permitted')
        weight = 'mt_center_weight' if view == 'identity' else 'mt_side_weight'
        add(stem + 'orient_weighted', 'mul', stem + 'orientDelta', weight)
    for family in FAMILIES:
        for view, _ in VIEWS:
            weight = 'mt_center_weight' if view == 'identity' else 'mt_side_weight'
            add(f'on95_{view}_{family}_weighted', 'mul', f'on95_{view}_{family}_mix', weight)
        add(f'on95_{family}_sides', 'add', f'on95_plus_{family}_weighted', f'on95_minus_{family}_weighted')
        add(f'on95_{family}_mix', 'add', f'on95_identity_{family}_weighted', f'on95_{family}_sides')
    add('on95_orientation_sides', 'add', 'on95_plus_orient_weighted', 'on95_minus_orient_weighted')
    add('on95_orientation_mix', 'add', 'on95_identity_orient_weighted', 'on95_orientation_sides')
    add('on95_orientation_column', 'reshape', 'on95_orientation_mix', shape=[-1, 1])
    add('on95_orientation_correction', 'pad', 'on95_orientation_column', 'zero', padding=[[0, 0], [9, 0]])
    for family in FAMILIES:
        if family in ROWS:
            row = ROWS[family]
            add(f'on95_{family}_row', 'slice', 'class_weights_v030', slices=[[row, row + 1], [None, None]])
        row_name = f'on95_{family}_row' if family in ROWS else 'sc80_body_row'
        add(f'on95_{family}_column', 'reshape', f'on95_{family}_mix', shape=[-1, 1])
        add(f'on95_{family}_correction', 'affine', f'on95_{family}_column', row_name, 'zero')
    add('on95_body_head_correction', 'add', 'on95_body_correction', 'on95_head_correction')
    add('on95_body_head_cup_correction', 'add', 'on95_body_head_correction', 'on95_cup_correction')
    add('on95_with_restoration_correction', 'add', 'on95_body_head_cup_correction', 'on95_restoration_correction')
    add('on95_correction', 'add', 'on95_with_restoration_correction', 'on95_orientation_correction')
    add('logits', 'add', 'on95_parent_logits', 'on95_correction')
    assert len(nodes) == ADDED_NODES, len(nodes)
    return nodes, copies


def construct():
    assert sha(Path(str(PREFIX) + '_spec.md')) == SPEC_SHA
    parent, weights, manifest = load(BASE, PARENT_MANIFEST_SHA)
    feature, feature_weights, fm = load(FEATURE, FEATURE_MANIFEST_SHA)
    assert fm['graph.json'] == '8e356ad9d90fa7417c2630a40890527cfd9d7c419dfc08c8371736907cccdf0d'
    assert fm['weights.npz'] == '05aa2ddffdd4cfb9a50c400b4d05dba320069012f0e089fdd38a5fc7a9e61689'
    inherited = json.loads((BASE / 'provenance.json').read_text())
    assert len(parent['nodes']) == PARENT_NODES and len(parent['feature_names']) == 302
    assert len(feature['nodes']) == 1236 and feature['inputs'] == list(FEATURE_INPUTS)
    assert set(feature_weights) == set(FEATURE_ARRAYS)
    for name, value in feature_weights.items():
        assert record(value) == record(weights[name]), name
    assert len(weights) == 555 and sum(v.size for v in weights.values()) == 2471932
    assert inherited['independent_numerical_specifications'] == sum(inherited['authored_specifications'].values()) == 1509
    assert not any('logits' in n['inputs'] for n in parent['nodes'])
    assert not any(k.startswith('on95_') for k in [*weights, *(n['name'] for n in parent['nodes'])])
    aliases = load_aliases(parent)
    extra, copies = suffix(feature, aliases)
    graph = deepcopy(parent)
    assert graph['nodes'][-1]['name'] == 'logits'
    graph['nodes'][-1]['name'] = 'on95_parent_logits'
    graph['nodes'].extend(extra)
    graph['outputs'] += EXTRA_OUTPUTS
    graph['open_nine_readout'] = dict(parent='v094', feature_manifest_sha256=FEATURE_MANIFEST_SHA,
        feature_copies=9, geometry_nodes_per_copy=1236,
        threshold_consensus='max(min(center,low),min(max(center,low),high))',
        orientation_pool='max(identityMedian,max(plusMedian,minusMedian))',
        branch_gate='sequential min: Qglobal,step(zero-cleanP),lowerDown,one-shortpath,one-localSupportedTwo',
        readout_only=True, all_parent_internal_computations_preserved=True,
        rows=ROWS, restoration_row='existing sc80_body_row',
        orientation_projection='reshape N1 then zero-pad nine columns on the left',
        correction_order='((((body+head)+cup)+restoration)+orientation')
    provenance = dict(candidate='v095', parent='v094', stage='direct authoring with development feedback',
        description='Nine unchanged ordered open-head walk copies with the single root-authored open-only readout residual.',
        rationale='Root supplied bounded development observations and fixed this composition prospectively. No population specificity follows; this author read no example, actual cue or feedback file.',
        prospective_spec_sha256=SPEC_SHA, source_audit_sha256=SOURCE_SHA, source_peer_audit_sha256=SOURCE_PEER_SHA,
        source_parent_manifest_sha256=PARENT_MANIFEST_SHA, source_parent_artifact_hashes=manifest,
        source_feature_manifest_sha256=FEATURE_MANIFEST_SHA, source_feature_artifact_hashes=fm,
        exact_feature_copies=copies, exact_readout_aliases={v+'_'+b:a for (v,b),a in aliases.items()},
        shared_feature_array_records={k:record(v) for k,v in feature_weights.items()},
        authored_specifications=inherited['authored_specifications'], independent_numerical_specifications=1509,
        expanded_tensor_elements=2471932, all_tensor_records=inherited['all_tensor_records'],
        graph_nodes=TOTAL_NODES, preserved_parent_nodes=PARENT_NODES, added_graph_nodes=ADDED_NODES,
        added_geometry_nodes=GEOMETRY_NODES, added_composition_nodes=ADDED_NODES-GEOMETRY_NODES,
        added_arrays=0, added_tensor_elements=0, added_independent_numerical_specifications=0,
        identity_feature_columns=302, parent_raw_archive_and_runner_unchanged=True,
        training_or_search=False, images_or_results_used_by_materializer=False,
        arithmetic_note='Explicit float32 residual added once to complete parent, preserving original internal body-to-waist, route, cup, orientation and seven qualifications. No reassociated parent rebuild.',
        verification_access='Authorized operational/source/spec/review artifacts, abstract scalar and nonglyph masks only; no examples, cue observations, result archives, sealed or confirmation files.',
        limitations='Ordered-walk feature accepts some spurs and discards alternate seeds. Pooling may transfer false witnesses into locally eligible branches. No accuracy, specificity, all-history or pretraining-exclusion claim.')
    info = dict(feature=feature, feature_weights=feature_weights, aliases=aliases, copies=copies, extra_outputs=EXTRA_OUTPUTS)
    return parent, graph, weights, provenance, info


def prepare():
    if STAGE.exists() or DEST.exists():
        raise RuntimeError('Preserve existing stage/candidate')
    try:
        parent, graph, weights, provenance, info = construct()
        verifier = Path(str(PREFIX) + '_verify.py')
        spec = importlib.util.spec_from_file_location('open_nine_head_v095_verify', verifier)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verification = module.verify(parent, graph, weights, provenance, info)
    except Exception:
        failure = Path(str(PREFIX) + '_failures.jsonl')
        with failure.open('a') as stream:
            stream.write(json.dumps(dict(stage='before_staging', error=traceback.format_exc())) + '\n')
        backup = Path(str(PREFIX) + f'_failed{len(failure.read_text().splitlines()):02d}')
        backup.mkdir()
        shutil.copy2(__file__, backup / 'author.py')
        if 'verifier' in locals() and verifier.exists():
            shutil.copy2(verifier, backup / 'source_verification.py')
        if 'graph' in locals():
            (backup / 'graph.json').write_text(json.dumps(graph, indent=2) + '\n')
            (backup / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        raise
    STAGE.mkdir(parents=True)
    for name, value in [('graph.json', graph), ('provenance.json', provenance), ('prefreeze_verification.json', verification)]:
        (STAGE / name).write_text(json.dumps(value, indent=2) + '\n')
    for source, name in [(BASE/'weights.npz','weights.npz'), (BASE/'runner.py','runner.py'),
                         (Path(__file__),'author.py'), (ROOT/'verify.py','verify.py'),
                         (verifier,'source_verification.py'), (Path(str(PREFIX)+'_spec.md'),'composition_spec.md')]:
        shutil.copy2(source, STAGE / name)
    assert sha(STAGE/'weights.npz') == sha(BASE/'weights.npz')
    hashes = {p.name:sha(p) for p in sorted(STAGE.iterdir()) if p.is_file()}
    (STAGE/'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(dict(status='STAGED', candidate='v095', nodes=TOTAL_NODES, features=302,
                         arrays=555, independent=1509, expanded=2471932, hashes=hashes), indent=2))


def freeze():
    if DEST.exists():
        raise RuntimeError('Frozen candidate exists; never overwrite')
    hashes = json.loads((STAGE/'SHA256SUMS.json').read_text())
    for name, digest in hashes.items():
        assert sha(STAGE/name) == digest, name
    assert sha(__file__) == hashes['author.py']
    gp = Path(str(PREFIX)+'_generic_verification.json')
    generic = json.loads(gp.read_text())
    assert generic['artifact_hashes'] == hashes and generic['nodes'] == TOTAL_NODES
    review = json.loads(Path(str(PREFIX)+'_peer_review.json').read_text())
    assert review['verdict'] == 'clean' and review['phase'] == 'staged_artifact'
    assert review['author_sha256'] == hashes['author.py']
    assert review['stage_manifest_sha256'] == sha(STAGE/'SHA256SUMS.json')
    assert review['generic_verification_sha256'] == sha(gp) and review['artifact_hashes'] == hashes
    ordinary = json.loads(Path(str(PREFIX)+'_ordinary_verification.json').read_text())
    assert ordinary['status'] == 'PASS' and ordinary['stage_manifest_sha256'] == sha(STAGE/'SHA256SUMS.json')
    shutil.copytree(STAGE, DEST)
    assert sha(DEST/'SHA256SUMS.json') == sha(STAGE/'SHA256SUMS.json')
    for name, digest in hashes.items():
        assert sha(DEST/name) == digest, name
    print(json.dumps(dict(status='FROZEN', candidate='v095', hashes=hashes,
                         stage_manifest_sha256=sha(STAGE/'SHA256SUMS.json')), indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','freeze'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else freeze()
