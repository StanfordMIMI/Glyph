import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request,urlopen

import pytest

from glyph.visualization.hosting import ReportServer,serve_html,serve_over_ssh


def test_root_opens_viewer_directly_and_other_files_are_not_served(tmp_path):
    page=tmp_path/'viewer.html';page.write_text('<html>First viewer</html>')
    other=tmp_path/'another.html';other.write_text('<html>Second viewer</html>')
    (tmp_path/'metadata.csv').write_text('private,data')
    with serve_html([page,other],open_browser=False) as server:
        with urlopen(server.url) as response:
            assert response.read()==page.read_bytes()
            assert response.headers['X-Glyph-Viewer']=='1'
        with urlopen(server.urls[1]) as response:assert response.read()==other.read_bytes()
        for suffix in ['metadata.csv','../metadata.csv','%2e%2e/metadata.csv']:
            with pytest.raises(HTTPError) as error:urlopen(server.url+suffix)
            assert error.value.code==404
        with urlopen(Request(server.url,method='HEAD')) as response:assert response.read()==b''
        page.write_text('<html>Updated viewer</html>')
        with urlopen(server.url) as response:assert b'Updated' in response.read()
    assert server._closed
    assert not server._thread.is_alive()


def test_local_open_uses_new_browser_tab(tmp_path,monkeypatch):
    import glyph.visualization.hosting as hosting
    monkeypatch.delenv('SSH_CONNECTION',raising=False)
    monkeypatch.delenv('SSH_TTY',raising=False)
    monkeypatch.setattr(hosting.sys,'platform','darwin')
    opened=[];monkeypatch.setattr(hosting.webbrowser,'open_new_tab',lambda url:opened.append(url))
    page=tmp_path/'viewer.html';page.write_text('<html>Viewer</html>')
    with serve_html(page) as server:assert opened==[server.url]


def test_remote_server_does_not_try_opening_remote_browser(tmp_path,monkeypatch):
    import glyph.visualization.hosting as hosting
    monkeypatch.setenv('SSH_CONNECTION','test ssh session')
    monkeypatch.setattr(hosting.webbrowser,'open_new_tab',lambda url:pytest.fail('remote browser was opened'))
    page=tmp_path/'viewer.html';page.write_text('<html>Viewer</html>')
    with serve_html(page) as server:assert server.url.startswith('http://127.0.0.1:')


def test_hosting_missing_html_is_clear(tmp_path):
    with pytest.raises(ValueError,match='No HTML reports'):ReportServer(tmp_path)
    with pytest.raises(ValueError,match='existing HTML'):ReportServer(tmp_path/'missing.html')


def test_ssh_client_waits_for_matching_viewer_and_opens_locally(tmp_path,monkeypatch):
    import glyph.visualization.hosting as hosting
    page=tmp_path/'a report.html';page.write_text('<html>SSH viewer</html>')
    commands=[];opened=[];servers=[]
    class FakeProcess:
        def __init__(self,args,stdin):
            commands.append(args)
            remote=shlex.split(args[-1])
            port=int(remote[remote.index('--port')+1]);session=remote[remote.index('--session-id')+1]
            server=ReportServer(page,port=port,session_id=session);servers.append(server)
            class Input(io.BytesIO):
                def close(self):server.close();super().close()
            self.stdin=Input();self.returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=0;return 0
    monkeypatch.setattr(subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(hosting,'_open_browser',lambda url:opened.append(url))
    with serve_over_ssh('rholland@torino','/remote/a report.html',remote_python='/env/bin/python',timeout=2) as server:
        assert opened==[server.url]
        assert commands[0][-2]=='rholland@torino'
        remote=shlex.split(commands[0][-1])
        assert remote[:3]==['/env/bin/python','-m','glyph']
        assert '/remote/a report.html' in remote
        assert '--stop-on-stdin-close' in remote
        with urlopen(server.url) as response:assert b'SSH viewer' in response.read()
    assert servers[0]._closed


def test_ssh_failure_does_not_open_browser(monkeypatch):
    import glyph.visualization.hosting as hosting
    class Exited:
        stdin=io.BytesIO()
        def poll(self):return 1
        def wait(self,timeout=None):return 1
    monkeypatch.setattr(subprocess,'Popen',lambda *args,**kwargs:Exited())
    monkeypatch.setattr(hosting,'_open_browser',lambda url:pytest.fail('Opened a failed viewer'))
    with pytest.raises(RuntimeError,match='SSH viewer exited'):
        serve_over_ssh('host','file.html',timeout=.1)


def test_serving_does_not_import_training_dependencies():
    result=subprocess.run([sys.executable,'-c',
        "from glyph.visualization.hosting import ReportServer; import sys; assert 'torch' not in sys.modules"],
        capture_output=True,text=True,env=os.environ.copy())
    assert result.returncode==0,result.stderr
