"""Resolve optional interpretation credentials without copying them into configs."""
import os


def setting(obj,name,default=None):
    return obj.get(name,default) if hasattr(obj,'get') else getattr(obj,name,default)


def connection(llm):
    provider=setting(llm,'provider','gemini_api')
    key=setting(llm,'gemini_api_key') or os.environ.get(setting(llm,'gemini_api_key_env','GEMINI_API_KEY')) or os.environ.get('GOOGLE_API_KEY')
    project=setting(llm,'vertex_project') or os.environ.get('GOOGLE_CLOUD_PROJECT')
    return provider,key,project


def ready(cfg):
    if not setting(cfg.autointerp,'enabled',True):return False
    provider,key,project=connection(cfg.autointerp.llm)
    return bool(project) if provider=='vertex' else bool(key) if provider in ['gemini','gemini_api'] else False
