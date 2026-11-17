"""在主服务器主机上把判官要看的文件（整片成片、切镜与镜对帧）写进判分 Cloud Storage 桶。

服务器的项目代理出口过不了 Gemini 地区闸（实测 400 ``User location is not supported``），但能写
``storage.googleapis.com``；所以字节在服务器写桶，本机再经本机代理把 ``gs://`` 对象登记给 Gemini。
本机经 ssh 标准输入送来一行 JSON：本文件源码与 ``ARGS``（1 小时有效的访问令牌、代理文件、桶、并发数、
``items``＝[{path, object, mime, size 或 null}]、写完后要删的远端目录 ``cleanup`` 或 null）；远端用 ``exec`` 跑，
只用标准库、不依赖远端检出。令牌只在进程内存里，不进命令行、磁盘与输出；代理取自项目 ``proxy.txt`` 的
``https_proxy``，不打印。输出一行 JSON：全部成功为 ``{"objects": [GCS 对象资源…]}``（与 items 同序），否则
``{"status": 码, "body": 正文}`` 并以非零码退出（``status`` 0 表示本地字节核对失败）。全部成功才删 ``cleanup``。
本机的调用方是 ``gemini_native.Storage.put_from_host``。
"""
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Failed(Exception):
    def __init__(self, status, body):
        super().__init__(status)
        self.status, self.body = status, body


def proxy_from(path: str) -> str:
    for line in open(path):
        name, _, value = line.strip().removeprefix('export ').partition('=')
        if name.strip() == 'https_proxy':
            return value.strip().strip('\'"')
    raise ValueError(f'{path}: no https_proxy line')


def main(args: dict) -> int:
    opener = build_opener(ProxyHandler({'https': proxy_from(args['proxy_file'])}), _NoRedirect())

    def put(item):
        size = os.path.getsize(item['path'])
        if item['size'] is not None and size != item['size']:
            raise Failed(0, f"{item['path']}: {size} bytes on host, expected {item['size']}")
        url = (f"https://storage.googleapis.com/upload/storage/v1/b/{args['bucket']}/o"
               f"?uploadType=media&name={quote(item['object'], safe='')}")
        with open(item['path'], 'rb') as handle:
            request = Request(url, data=handle.read(), method='POST',
                              headers={'Authorization': 'Bearer ' + args['token'], 'Content-Type': item['mime']})
        try:
            with opener.open(request, timeout=args['timeout']) as response:
                stored = json.loads(response.read())
        except HTTPError as error:
            raise Failed(error.code, error.read().decode(errors='replace').replace(args['token'], '[REDACTED]')[:2000])
        if int(stored['size']) != size:
            raise Failed(0, f"{item['object']}: stored {stored['size']} of {size} bytes")
        return {k: stored[k] for k in ('bucket', 'name', 'size', 'md5Hash', 'generation')}

    try:
        with ThreadPoolExecutor(args['workers']) as pool:
            objects = list(pool.map(put, args['items']))
    except Failed as error:
        print(json.dumps({'status': error.status, 'body': error.body}))
        return 3
    if args['cleanup']:
        shutil.rmtree(args['cleanup'])
    print(json.dumps({'objects': objects}))
    return 0


if __name__ == '__host__':
    sys.exit(main(ARGS))  # noqa: F821 — ARGS 由本机经标准输入送来
