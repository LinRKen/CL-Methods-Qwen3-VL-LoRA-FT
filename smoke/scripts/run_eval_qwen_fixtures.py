#!/usr/bin/env python3
"""Run 30 predeclared semantic judge cases using production functions and a selected local judge."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'evaluation/COSINE'))
import eval_cosine as cli
from dataset_utils import read_jsonl, write_json_atomic
from qwen_judge import build_messages, scored_record


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',required=True); parser.add_argument('--config',type=Path,default=ROOT/'smoke/configs/eval_qwen_4b.json'); opts=parser.parse_args()
    config=json.loads(opts.config.resolve().read_text())
    os.environ['CUDA_VISIBLE_DEVICES']=config['gpu_uuid']
    args=SimpleNamespace(**{key:config[key] for key in ['batch_size','tensor_parallel_size','gpu_memory_utilization','max_model_len','max_new_tokens','repair_attempts','seed','dtype']})
    args.dataset='smoke_handwritten_30'
    devices=cli.verify_gpu_selection(args)
    assert devices==[dict(physical_index=1,uuid=config['gpu_uuid'],pci_bus_id=devices[0]['pci_bus_id'])]
    fixture=ROOT/'smoke/fixtures/eval_qwen_cases.json'; cases=json.loads(fixture.read_text())
    records=[{k:v for k,v in row.items() if k!='expected'} for row in cases]
    output=Path(opts.output_dir); output.mkdir(parents=True,exist_ok=False)
    model=Path(config['judge_model'])
    run,fingerprint=cli.make_run(args,fixture.resolve(),fixture.resolve(),model)
    processor=cli.load_processor(model)
    prompts={row['inference_index']:cli.make_prompt(processor,build_messages(row),args,row['inference_index']) for row in records}
    lengths=[len(processor.tokenizer(p,add_special_tokens=False)['input_ids']) for p in prompts.values()]
    engine,sampling=cli.load_engine(args,model); done={}
    try:
        with (output/'judge.raw.jsonl').open('x') as handle:
            for start in range(0,len(records),args.batch_size):
                cli.judge_batch(engine,sampling,processor,records[start:start+args.batch_size],prompts,args,handle,done,run,fingerprint,devices,len(records))
    finally:
        cli.shutdown_engine(engine)
    scored=[scored_record(row,done[row['inference_index']]['parsed'],done[row['inference_index']]['parse_error']) for row in records]
    self_check=cli.restore_raw(output/'judge.raw.jsonl',records,run,fingerprint)
    assert self_check==done
    comparisons=[]
    for case,row in zip(cases,scored):
        expected=case['expected']; actual={'status':row['status'],'sample_score':row['sample_score']}
        passed=actual['status']==expected['status'] and abs(actual['sample_score']-expected['sample_score'])<1e-9
        if 'numbers' in expected:
            actual['numbers']=[row['gt'],row['pred']]
            passed=passed and actual['numbers']==expected['numbers']
        comparisons.append({'case_id':case['case_id'],'passed':passed,'expected':expected,'actual':actual,
                            'attempts':len(done[case['inference_index']]['attempts'])})
    summary={'total':len(cases),'passed':sum(row['passed'] for row in comparisons),
             'parse_failures':sum(row['judge_parse_error'] is not None for row in scored),
             'repairs':sum(len(row['attempts'])>1 for row in done.values()),'prompt_tokens_min':min(lengths),
             'prompt_tokens_max':max(lengths),'devices':devices,'config':str(opts.config.resolve()),'run':run,'cases':comparisons}
    write_json_atomic(output/'judge.scored.jsonl',scored,jsonl=True)
    write_json_atomic(output/'fixture.metrics.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if summary['passed']==summary['total'] else 2

if __name__=='__main__': sys.exit(main())
