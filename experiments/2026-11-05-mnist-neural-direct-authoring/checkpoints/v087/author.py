"""Exact root-fixed complete v086 plus v085 closed-three correction; no data/fitting."""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
if ROOT.name=='v087' and ROOT.parent.name=='candidates':ROOT=ROOT.parent.parent
BASE=ROOT/'candidates/v086';SOURCE=ROOT/'candidates/v085';DEST=ROOT/'candidates/v087'
PREFIX=ROOT/'contributions/composition_v087';STAGE=ROOT/'runtime/composition_v087_staging'
SPEC_SHA='6629958cfe3097da3e42f8ac569c34d9514fec630fdc264d3de83c642a6db4c9'
PINNED={'v079':'bc344d96440a1dcbba51a4eb4b2e03acc33d90636eddeef10f6ba362d6405a6c',
        'v085':'315ef6b87361308ffef618d82cb628937463917dae1043464ed80c5134945e73',
        'v086':'c6778dcb942f95c033c7d42d880f6bd8375db3c7924cdba8f2968aa58e5d02cf'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(value):
    return dict(dtype=str(value.dtype),shape=list(value.shape),entries=value.size,
                raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def load(version):
    path=ROOT/'candidates'/version;manifest=json.loads((path/'SHA256SUMS.json').read_text())
    hashes={name:sha(path/name) for name in manifest}
    assert hashes==manifest and hashes['graph.json']==PINNED[version]
    with np.load(path/'weights.npz',allow_pickle=False) as archive:weights={k:archive[k] for k in archive.files}
    return dict(graph=json.loads((path/'graph.json').read_text()),weights=weights,
                provenance=json.loads((path/'provenance.json').read_text()),hashes=hashes)


def closure(graph,roots,boundary):
    needed=set(roots);kept=[]
    for n in reversed(graph['nodes']):
        if n['name'] in needed and n['name'] not in boundary:
            assert not any('logits' in k for k in [n['name'],*n['inputs']])
            kept.append(deepcopy(n));needed.update(n['inputs'])
    return kept[::-1]


def construct():
    assert sha(Path(str(PREFIX)+'_spec.md'))==SPEC_SHA
    base,source,common=load('v086'),load('v085'),load('v079')
    parent,weights,previous=base['graph'],base['weights'],base['provenance']
    assert len(weights)==552 and sum(v.size for v in weights.values())==1242619
    assert set(source['weights'])==set(common['weights'])==set(weights)
    for other in [source,common]:
        for name,value in other['weights'].items():assert record(value)==record(weights[name]),name
        for name in ['weights.npz','runner.py']:assert other['hashes'][name]==base['hashes'][name]
    assert previous['independent_numerical_specifications']==sum(previous['authored_specifications'].values())==1506
    expected_rows={196:[0,0,0,4,0,0,0,0,0,0],62:[0,0,0,0,0,0,0,0,6,-3],121:[0,0,0,0,0,0,0,-3,0,5],
                   187:[0,0,0,0,0,0,0,-3,0,3],209:[0,0,0,0,0,0,0,2,0,0],210:[0,0,0,0,0,0,0,4,0,0]}
    for index,row in expected_rows.items():np.testing.assert_array_equal(weights['class_weights_v030'][index],np.asarray(row,np.float32))
    np.testing.assert_array_equal(weights['pr_class_weights'],np.asarray([[0,0,0,0,0,0,0,-3,0,0]],np.float32))
    pnodes={n['name']:n for n in parent['nodes']};snodes={n['name']:n for n in source['graph']['nodes']}
    for n in common['graph']['nodes']:
        if n['name']!='logits':assert n==pnodes[n['name']]==snodes[n['name']],n['name']
    for key in (set(pnodes)&set(snodes))-{'logits'}:assert pnodes[key]==snodes[key],key
    assert pnodes['logits']==dict(name='logits',op='add',inputs=['co86_parent_logits','sc80_correction'])
    assert pnodes['co86_parent_logits']==dict(name='co86_parent_logits',op='add',inputs=['fl84_parent_logits','fl84_correction'])
    assert pnodes['fl84_parent_logits']==dict(name='fl84_parent_logits',op='add',inputs=['co79_uncalibrated_logits','to78_class_bias'])
    assert not any('logits' in n['inputs'] for n in parent['nodes'])
    graph=deepcopy(parent)
    for n in graph['nodes']:
        if n['name']=='logits':n['name']='co87_parent_logits'
    boundary={'pixels',*weights,*(n['name'] for n in graph['nodes'])}
    body=closure(source['graph'],['cp85_correction'],boundary);assert len(body)==20
    graph['nodes'].extend(body)
    graph['nodes'].append(dict(name='logits',op='add',inputs=['co87_parent_logits','cp85_correction']))
    diagnostics=['cp85_'+b+'_'+name for b in ['center','low','high'] for name in ['permission','permitted_raw','new','delta','weighted']]
    diagnostics += ['cp85_delta_mix']
    graph['outputs']=parent['outputs']+['co87_parent_logits','cp85_correction']+diagnostics
    graph['diagnostic_scalars']=parent.get('diagnostic_scalars',[])+diagnostics
    known={'pixels',*weights}
    for n in graph['nodes']:
        assert n['name'] not in known and set(n['inputs'])<=known,n['name'];known.add(n['name'])
    assert len(graph['outputs'])==len(set(graph['outputs']))
    for key in ['outputs','feature_names','diagnostic_maps','diagnostic_scalars']:assert set(graph.get(key,[]))<=known
    assert len(parent['nodes'])==37422 and len(graph['nodes'])==37443 and len(graph['feature_names'])==302
    provenance={k:deepcopy(previous[k]) for k in ['authored_specifications','independent_numerical_specifications','expanded_tensor_elements']}
    provenance.update(candidate='v087',parent='v086',component='v085 exact cp85_correction',excluded=['v081','v082','rotation'],
        stage='direct authoring with development feedback',
        description='Complete v086 plus exact v085 cp85_correction once, preserving source old-floor, max-before-mixture, class3-only projection and association.',
        rationale='Root selected this single composition using supplied score metadata. No numerical choices, variants, additional routes, fitting or new example access.',
        declaration_sha256=SPEC_SHA,source_hashes={'v086':base['hashes'],'v085':source['hashes'],'v079':common['hashes']},
        exact_readout='COMPLETE_v086_logits + cp85_correction',
        exact_component='(delta_center*.5+(delta_low*.25+delta_high*.25))[:,None] @ row196 + zero; delta=max(actual_old,raw*(1-Qoutside))-actual_old',
        tensor_count=len(weights),new_independent_numerical_specifications=0,new_expanded_tensor_elements=0,new_array_count=0,
        preserved_parent_tensor_records={name:record(value) for name,value in weights.items()},
        raw_weights_byte_identical_to_parent=True,runner_byte_identical_to_parent=True,
        copied_closure_root='cp85_correction',copied_closure_count=len(body),closure_range=[len(parent['nodes']),len(parent['nodes'])+len(body)],
        new_graph_nodes=len(graph['nodes'])-len(parent['nodes']),parent_feature_columns_unchanged=True,parent_feature_columns=302,
        common_v079_nodes_exact=True,source_rows=expected_rows,source_projection_row='class_weights_v030[196:197]: +4 class3 only',
        count_note='552 arrays,1506 independent specifications,1242619 expanded entries unchanged.20 exact live source nodes and one final addition; no new numerical values.',
        root_provided_context='v0859491 versus v0799490; one corrected prediction, no lost correct predictions,140 class3 logit changes with other columns unchanged. v086 validation pending at declaration. Metadata from root only; no archive read; gains may overlap.',
        source_limitations=source['provenance'].get('limits'),parent_limitations=previous.get('limitations'),
        limitations='New three support may regress other digits and interact with parent three/seven/eight/nine competition. Gains need not add. No accuracy or sealed-success claim.',
        source_failure_records='All source artifacts and prior failure records remain preserved without modification or result reopening.',
        training_or_search=False,verification_access='Source artifacts and generic fields/scalars only; no MNIST, labels, result archives, actual-example heads, scoring or sealed data.')
    return parent,graph,weights,provenance,dict(base=base,source=source,common=common,body=body)


def prepare():
    if STAGE.exists() or DEST.exists():raise RuntimeError('Preserve existing stage/candidate')
    try:
        parent,graph,weights,provenance,info=construct()
        verifier=Path(str(PREFIX)+'_verify.py');spec=importlib.util.spec_from_file_location('composition_v087_verify',verifier)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        verification=module.verify(parent,graph,weights,provenance,info)
    except Exception:
        failure=Path(str(PREFIX)+'_failures.jsonl')
        with failure.open('a') as stream:stream.write(json.dumps(dict(stage='before_staging',error=traceback.format_exc()))+'\n')
        index=len(failure.read_text().splitlines())
        shutil.copy2(__file__,Path(str(PREFIX)+f'_failed_{index}_author.py'))
        verifier=Path(str(PREFIX)+'_verify.py')
        if verifier.exists():shutil.copy2(verifier,Path(str(PREFIX)+f'_failed_{index}_verify.py'))
        raise
    STAGE.mkdir(parents=True)
    for name,value in [('graph.json',graph),('provenance.json',provenance),('prefreeze_verification.json',verification)]:
        (STAGE/name).write_text(json.dumps(value,indent=2)+'\n')
    for src,name in [(BASE/'weights.npz','weights.npz'),(BASE/'runner.py','runner.py'),(Path(__file__),'author.py'),
                     (ROOT/'verify.py','verify.py'),(verifier,'source_verification.py'),(Path(str(PREFIX)+'_spec.md'),'composition_spec.md')]:shutil.copy2(src,STAGE/name)
    assert sha(STAGE/'weights.npz')==sha(BASE/'weights.npz') and sha(STAGE/'runner.py')==sha(BASE/'runner.py')
    hashes={p.name:sha(p) for p in sorted(STAGE.iterdir()) if p.is_file()}
    (STAGE/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(status='STAGED',candidate='v087',nodes=len(graph['nodes']),features=302,arrays=552,
                         independent=1506,expanded=1242619,hashes=hashes),indent=2))


def freeze():
    if DEST.exists():raise RuntimeError('Frozen candidate exists; never overwrite')
    hashes=json.loads((STAGE/'SHA256SUMS.json').read_text())
    for name,digest in hashes.items():assert sha(STAGE/name)==digest,name
    assert sha(__file__)==hashes['author.py']
    generic_path=Path(str(PREFIX)+'_generic_verification.json');generic=json.loads(generic_path.read_text())
    assert generic['artifact_hashes']==hashes and generic['nodes']==37443
    review=json.loads(Path(str(PREFIX)+'_review.json').read_text())
    assert review['verdict']=='clean' and review['phase']=='staged_artifact'
    assert review['author_sha256']==hashes['author.py'] and review['stage_manifest_sha256']==sha(STAGE/'SHA256SUMS.json')
    assert review['generic_verification_sha256']==sha(generic_path) and review['artifact_hashes']==hashes
    shutil.copytree(STAGE,DEST)
    assert sha(DEST/'SHA256SUMS.json')==sha(STAGE/'SHA256SUMS.json')
    for name,digest in hashes.items():assert sha(DEST/name)==digest,name
    print(json.dumps(dict(status='FROZEN',candidate='v087',hashes=hashes,
                         stage_manifest_sha256=sha(STAGE/'SHA256SUMS.json')),indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','freeze'])
    args=parser.parse_args();prepare() if args.action=='prepare' else freeze()
