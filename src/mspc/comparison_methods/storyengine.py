"""官方 StoryEngine 的本地 Wan 端口与可恢复的单故事执行。

故事规划、参考、grounding、修复及选择仍由固定上游持有；本模块只接后端。
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def load_config(path):
    from mspc.config.shared_root import expand_shared_root_token
    from mspc.benchmark.generation import load_config as load_generation_config
    config=json.loads(expand_shared_root_token(Path(path).read_text(),str(path)))
    code_root=Path(__file__).resolve().parents[3]
    native_path=code_root/config['native_wan_config']
    native=load_generation_config(native_path)['methods'][config['native_wan_method']]
    native['upstream']=str(code_root/'third_party/wan22_runtime')
    if native['checkpoint'] is not None or native['model']['memory_prefix'] is not None:
        raise ValueError('StoryEngine requires the public native Wan without a checkpoint or memory extension')
    config['native_wan_settings']=native
    return config


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def prepare(config):
    """只消费冻结原请求及公开交付元数据，不读取共同分镜的隐藏状态。"""
    sys.path.insert(0,config['storyengine']['upstream'])
    from story_engine.config import AppConfig
    from story_engine.domain.request import ProjectRequest
    source=read_json(config['raw_request_source'])
    story=next(s for s in source['stories'] if s['story_id']==config['story_id'])
    if story['split']!='dev':raise ValueError('only one frozen development story is authorized')
    original=story['inputs']['shots']
    count=original['expected_shot_count']
    if count is None:raise ValueError('shot_target must be explicit in the original request')
    request=ProjectRequest.model_validate({
        'task_id':story['story_id'],
        'idea':original['text']+f"\nDELIVERY: approximately {story['target_seconds']} seconds total; {count} shots; 16 fps, no audio.",
        'shot_target':count,'visual_style':'Follow the visual style of the original request.',
        'resolution':{'width':832,'height':480},'provided_assets':[],
        'generation_requirements':{'output_language':'English','delivery_requirements':{
            'allowed_video_duration_seconds':list(range(1,15)),
            'video_resolution':{'width':832,'height':480},'video_fps':16,'audio':False}}})
    endpoints=config['storyengine']['provider_endpoints']
    models=config['storyengine']['provider_models']
    def spec(adapter,role,options):
        return {'adapter':adapter,'model':models[role],'base_url':endpoints[role],
                'timeout_seconds':600,'options':options}
    app=AppConfig.model_validate({'project':{'output_root':config['output_root']},'providers':{
        'planner':spec('openai_responses','planner',{'max_attachments':16,
            'preflight_mode':config['storyengine']['model_discovery']['planner']}),
        'judge':{'adapter':'gemini_native','model':models['judge'],
            'base_url':endpoints['judge'],'timeout_seconds':600,
            'options':{'credential_mode':'bearer','judge_max_media':16,
                'preflight_mode':config['storyengine']['model_discovery']['judge']}},
        'image':spec('openai_images','image',{'preflight_mode':config['storyengine']['model_discovery']['image'],
            'quality':'low','max_input_images':16,
            'supported_aspect_ratios':['26:15','2:1'],'supported_resolutions':[
                {'width':1664,'height':960},{'width':1536,'height':768}]}),
        'video':{'adapter':'local_wan_first_frame','model':'Wan2.2-TI2V-5B'}},
        'generation':{'attempts':3,'candidates_per_attempt':1,'story_planning_attempts':3,
                      'grounding_attempts':3,'provider_retries':0,'max_concurrency':2}})
    root=Path(config['output_root'])/'inputs'
    for path,value in [(root/'storyengine-request.json',request.model_dump(mode='json')),
                       (root/'storyengine-runtime.json',app.redacted_dict())]:
        if path.exists() and read_json(path)!=value:raise FileExistsError('refusing to change frozen pilot input: '+str(path))
        write_json(path,value)
    write_json(root/'storyengine-input-provenance.json',{'source':config['raw_request_source'],
        'source_field':'inputs.shots.text','story_id':story['story_id'],'scope':'system',
        'shared_delivery_metadata':{'shot_count':count,'target_seconds':story['target_seconds'],'fps':16},
        'source_storyboard_or_source_frames_supplied':False})
    print(json.dumps({'story_id':story['story_id'],'shot_target':count,'status':'input_prepared'}))


def compose(config):
    sys.path.insert(0, config['storyengine']['upstream'])
    from PIL import Image
    from story_engine.config import AppConfig, SecretValue
    from story_engine.domain.request import ProjectRequest, PixelSize
    from story_engine.errors import ProviderError, ProviderErrorKind
    from story_engine.providers.adapters.openai import OpenAIResponsesAdapter, OpenAIImageAdapter
    from story_engine.providers.adapters.gemini import GeminiGenerateContentAdapter
    from story_engine.providers.ports import (ImageResult, MediaPayload, ProviderCallMetrics,
        ProviderIdentity, ProviderCapabilityProfile, VideoCapabilities, VideoJob, VideoJobStatus, VideoResult)
    from story_engine.providers.registry import ProviderSet
    from story_engine.run_state import RunStore
    from story_engine.workflow import Workflow
    from story_engine.providers.transport import HttpTransport

    settings=config['storyengine']
    request=ProjectRequest.model_validate(read_json(Path(config['output_root'])/'inputs/storyengine-request.json'))
    app=AppConfig.model_validate(read_json(Path(config['output_root'])/'inputs/storyengine-runtime.json'))
    run_dir=Path(config['output_root'])/'inference/raw/storyengine'/config['story_id']
    persisted=run_dir/'config.redacted.json'
    if persisted.exists() and read_json(persisted)!=app.redacted_dict():
        raise ValueError('resume requires the identical recorded StoryEngine runtime config')
    store=RunStore(run_dir)
    store.initialize(run_id=config['story_id'], request_hash=request.request_hash,
                     request=request,redacted_config=app.redacted_dict())
    ledger_path=Path(config['output_root'])/'metrics/storyengine-provider-usage.json'
    ledger=read_json(ledger_path) if ledger_path.exists() else {'calls':[], 'estimated_usd':0.0,
        'pricing_basis':'official token rates; gateway billed amount must be verified separately',
        'limit_usd':settings['api_limit_usd']}
    budget_lock=asyncio.Lock()

    class MeteredTransport(HttpTransport):
        def __init__(self, role, *args, **kwargs):
            super().__init__(*args,**kwargs)
            self.role=role

        async def request(self, method, path_or_url, **kwargs):
            if method.upper()!='POST':
                return await super().request(method,path_or_url,**kwargs)
            async with budget_lock:
                model=getattr(app.providers,self.role).model
                body=kwargs.get('json_body') or kwargs.get('data') or {}
                if self.role=='planner' and kwargs.get('json_body') is not None:
                    body={**body,'max_output_tokens':settings['max_output_tokens']}
                    kwargs['json_body']=body
                elif self.role=='judge' and kwargs.get('json_body') is not None:
                    body={**body,'generationConfig':{**body.get('generationConfig',{}),
                          'maxOutputTokens':settings['max_output_tokens']}}
                    kwargs['json_body']=body
                reservation=settings['maximum_call_reserve_usd'][self.role]
                if ledger['estimated_usd']+reservation>ledger['limit_usd']:
                    raise ProviderError(ProviderErrorKind.UNKNOWN,'pilot API budget reached',retryable=True)
                row={'role':self.role,'model':model,'endpoint_path':path_or_url,
                     'status':'submitted','reserved_usd':reservation}
                ledger['calls'].append(row)
                write_json(ledger_path,ledger)
                try:
                    result=await super().request(method,path_or_url,**kwargs)
                except BaseException as error:
                    row['status']='outcome_unknown_or_rejected'
                    if isinstance(error,ProviderError):
                        row.update(http_status=error.status_code,error_kind=error.kind.value,
                                   error_message=error.message)
                    ledger['estimated_usd']+=reservation
                    write_json(ledger_path,ledger)
                    if isinstance(error,ProviderError) and error.status_code in (400,401,403,404):
                        # 配置／接口拒收不消耗三次图像修复；保留请求并中断等待接线修正。
                        raise ProviderError(error.kind,error.message,retryable=True,
                                            status_code=error.status_code,details=error.details) from None
                    raise
                payload=result.response.json()
                usage=payload.get('usage',payload.get('usageMetadata',{}))
                row.update(status='returned',usage=usage,returned_model=payload.get('model'),
                           elapsed_seconds=result.elapsed_seconds,request_id=payload.get('id'))
                rates=settings['token_rates_usd_per_million'][self.role]
                if self.role=='image':
                    details=usage.get('input_tokens_details',{})
                    text=details.get('text_tokens',0);images=details.get('image_tokens',0)
                    output=usage.get('output_tokens',0)
                    estimate=(text*rates['text_input']+images*rates['image_input']+output*rates['output'])/1e6
                elif self.role=='judge':
                    estimate=(usage.get('promptTokenCount',0)*rates['input']+
                              (usage.get('candidatesTokenCount',0)+usage.get('thoughtsTokenCount',0))*rates['output'])/1e6
                else:
                    estimate=(usage.get('input_tokens',0)*rates['input']+usage.get('output_tokens',0)*rates['output'])/1e6
                if not usage:
                    row['usage_missing']=True
                    estimate=reservation
                row['estimated_usd']=estimate
                row['provider_cost_fields']={k:payload[k] for k in ('cost','cost_usd','billed_cost') if k in payload}
                ledger['estimated_usd']+=estimate
                write_json(ledger_path,ledger)
                return result

    def transport(role):
        spec=getattr(app.providers,role)
        credential=SecretValue(Path(settings['credential_files'][role]).read_text().strip())
        return MeteredTransport(role,base_url=spec.base_url,credential=credential,
            timeout_seconds=spec.timeout_seconds,max_retries=0)

    class ResizedImageProvider(OpenAIImageAdapter):
        @property
        def capability_profile(self):
            image=self._profile.image.model_copy(update={'supported_resolutions':tuple(
                PixelSize(**p) for p in settings['logical_image_resolutions'])})
            return self._profile.model_copy(update={'image':image})

        async def generate_image(self, req):
            logical=req.resolution
            service=PixelSize(width=1664,height=960) if (logical.width,logical.height)==(832,480) else logical
            result=await super().generate_image(req.model_copy(update={'resolution':service}))
            raw=Path(config['output_root'])/'inference/raw/images/storyengine-service'
            raw.mkdir(parents=True,exist_ok=True)
            name=f'image-{len(list(raw.glob("image-*.png")))+1:04d}.png'
            (raw/name).write_bytes(result.payload.data)
            if service!=logical:
                image=Image.open(io.BytesIO(result.payload.data)).convert('RGB')
                if image.size!=(service.width,service.height):
                    raise ValueError('image service returned an undeclared size')
                image=image.resize((logical.width,logical.height),Image.Resampling.LANCZOS)
                buffer=io.BytesIO();image.save(buffer,format='PNG')
                result=ImageResult(payload=MediaPayload(data=buffer.getvalue(),media_type='image/png'),metrics=result.metrics)
            return result

    class LocalWanProvider:
        def __init__(self):
            self.jobs=Path(config['output_root'])/'inference/raw/storyengine-wan-jobs'
            self.jobs.mkdir(parents=True,exist_ok=True)
            self._profile=ProviderCapabilityProfile(
                identity=ProviderIdentity(adapter='local_wan_first_frame',model='Wan2.2-TI2V-5B',
                    endpoint='local process on a separately allocated GPU',adapter_version='1'),
                video=VideoCapabilities(supports_start_image=True,max_reference_images=0,
                    supported_durations=tuple(range(1,15)),supported_resolutions=(PixelSize(width=832,height=480),),
                    supported_fps=(16,),max_prompt_characters=32000,poll_interval_seconds=1,
                    job_timeout_seconds=3600,submit_is_idempotent=False))

        @property
        def capability_profile(self):return self._profile

        async def preflight(self):
            for path in config['native_wan_settings']['weights'].values():
                for item in path if isinstance(path,list) else [path]:
                    if not Path(item).exists():raise FileNotFoundError(item)

        async def aclose(self):pass

        def job(self,name,status,elapsed=0):
            return VideoJob(job_id=name,status=status,metrics=ProviderCallMetrics(
                call_ref=name+'-'+status.value,elapsed_seconds=elapsed,cost_usd=0))

        async def submit_video(self,req):
            if req.reference_images:raise ValueError('native Wan only accepts the declared start image')
            name=f'wan-video-{len(list(self.jobs.glob("wan-video-*")))+1:04d}'
            target=self.jobs/name
            target.mkdir()
            write_json(target/'request.json',{'typed_request':req.model_dump(mode='json'),
                'first_frame_path':str(store.artifacts.verify(req.start_image.artifact_ref)),
                'seed':config['seed']+(int(name.rsplit('-',1)[1])-1)*1009,
                'frames':req.duration*req.fps+1})
            return self.job(name,VideoJobStatus.QUEUED)

        async def poll_video(self,name):
            path=self.jobs/name/'result.json'
            if not path.exists():
                raise ProviderError(ProviderErrorKind.UNKNOWN,
                    'local video request persisted; resume after an idle GPU is allocated and the shot is rendered',retryable=True)
            result=read_json(path)
            if result['status']!='completed':
                return self.job(name,VideoJobStatus.FAILED,result['elapsed_seconds'])
            return self.job(name,VideoJobStatus.COMPLETED)

        async def download_video(self,name):
            result=read_json(self.jobs/name/'result.json')
            return VideoResult(payload=MediaPayload(data=Path(result['video_path']).read_bytes(),media_type='video/mp4'),
                metrics=ProviderCallMetrics(call_ref=name+'-download',elapsed_seconds=result['elapsed_seconds'],cost_usd=0))

    credentials={role:SecretValue(Path(settings['credential_files'][role]).read_text().strip())
                 for role in ['planner','judge','image']}
    planner=OpenAIResponsesAdapter(app.providers.planner,credentials['planner'],store.artifacts,max_retries=0)
    judge=GeminiGenerateContentAdapter(app.providers.judge,credentials['judge'],store.artifacts,max_retries=0)
    image=ResizedImageProvider(app.providers.image,credentials['image'],store.artifacts,max_retries=0)
    for role,provider in [('planner',planner),('judge',judge),('image',image)]:
        provider._transport=transport(role)
    providers=ProviderSet(planner=planner,image=image,judge=judge,video=LocalWanProvider())
    workflow=Workflow(request=request,config=app,run_store=store,providers=providers)
    return workflow


def render_wan_job(config,job_id):
    import torch
    from PIL import Image
    from mspc.comparison_methods.renderer import load_backend
    from mspc.data.shot_text import load_umt5_token_counter
    output=Path(config['output_root'])/'inference/raw/storyengine-wan-jobs'/job_id
    if (output/'result.json').exists():raise FileExistsError('the native shot has an existing outcome')
    request=read_json(output/'request.json')
    settings=config['native_wan_settings']
    sys.path.insert(0,settings['upstream'])
    counter=load_umt5_token_counter(Path(settings['weights']['tokenizer']))
    tokens=counter(request['typed_request']['prompt'])
    if tokens>512:raise ValueError(f'Wan prompt has {tokens} tokens; no silent truncation')
    began=time.monotonic()
    runtime,_=load_backend(settings,request['typed_request']['fps'])
    load_seconds=time.monotonic()-began
    image=Image.open(request['first_frame_path']).convert('RGB')
    latent=runtime.encode_single_frame(image)
    frames=request['frames']
    runtime.generate_case(prompts=[request['typed_request']['prompt']],shot_groups=[[0,frames]],
        latent_shot_groups=[[0,(frames-1)//4+1]],shot_frame_counts=[frames],output_path=output/'video.mp4',
        seed=request['seed'],first_frame=latent,first_frame_pin_until_sigma=0.0)
    write_json(output/'result.json',{'status':'completed','video_path':str(output/'video.mp4'),
        'elapsed_seconds':time.monotonic()-began,'load_seconds':load_seconds,'frames':frames,
        'tokens':tokens,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','validate','run','wan-shot'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--job-id')
    args=parser.parse_args()
    config=load_config(args.config)
    if args.command=='prepare':prepare(config);return
    if args.command=='wan-shot':render_wan_job(config,args.job_id);return
    workflow=compose(config)
    if args.command=='validate':
        workflow._validate_declared_capabilities()
        print(json.dumps({'story_id':config['story_id'],'status':'offline_contract_valid'}))
        asyncio.run(workflow.providers.aclose())
    else:
        state=asyncio.run(workflow.execute())
        print(json.dumps({'story_id':config['story_id'],'status':state.status.value}))


if __name__=='__main__':main()
