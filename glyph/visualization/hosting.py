"""Optional loopback hosting for self-contained Glyph reports."""
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlsplit
import webbrowser


def _report_files(paths):
    if isinstance(paths, (str, Path)):
        path = Path(paths).expanduser().resolve()
        paths = sorted(path.glob('*.html')) if path.is_dir() else [path]
    files = list(dict.fromkeys(Path(p).expanduser().resolve() for p in paths))
    if not files:
        raise ValueError('No HTML reports found. Generate one with glyph.visualize() first.')
    for path in files:
        if not path.is_file() or path.suffix.lower() != '.html':
            raise ValueError(f'Expected an existing HTML report: {path}')
    return files


def _open_browser(url):
    remote = bool(os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_TTY'))
    headless = sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    if remote or headless:
        print('Remote/headless session: open the displayed URL through a forwarded port.', flush=True)
        return False
    return webbrowser.open_new_tab(url)


class ReportServer:
    def __init__(self, paths, port=0, session_id=None):
        import uuid
        self.session_id = session_id or uuid.uuid4().hex
        session = self.session_id
        self.files = _report_files(paths)
        routes = {f'/report/{i}/{p.name}': p for i,p in enumerate(self.files)}
        files = self.files

        class Handler(BaseHTTPRequestHandler):
            server_version = 'GlyphViewer/1'
            def _respond(self, body=True):
                route = unquote(urlsplit(self.path).path)
                path = files[0] if route == '/' else routes.get(route)
                try:
                    content = path.read_bytes() if path else None
                except OSError:
                    content = None
                if content is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Length',str(len(content)))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Glyph-Viewer','1')
                self.send_header('X-Glyph-Session',session)
                self.end_headers()
                if body:self.wfile.write(content)
            def do_GET(self):self._respond()
            def do_HEAD(self):self._respond(False)
            def log_message(self,*args):pass

        self._server = ThreadingHTTPServer(('127.0.0.1',int(port)),Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_port
        self.url = f'http://127.0.0.1:{self.port}/'
        self.urls = [self.url.rstrip('/')+quote(route) for route in routes]
        self._thread = threading.Thread(target=self._server.serve_forever,daemon=True)
        self._thread.start()
        self._closed = False

    def close(self):
        if self._closed:return
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=3)
        self._closed = True

    def wait(self):
        try:
            while self._thread.is_alive():self._thread.join(timeout=1)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def serve_html(paths, *, port=0, open_browser=True, session_id=None):
    server = ReportServer(paths,port=port,session_id=session_id)
    print(f'Glyph viewer: {server.url}',flush=True)
    if open_browser:_open_browser(server.url)
    return server


class RemoteReportServer:
    def __init__(self, host, path, *, port=0, open_browser=True, remote_python=None, timeout=60):
        import shlex
        import socket
        import subprocess
        import time
        from urllib.request import Request, urlopen
        import uuid
        if not host or host.startswith('-'):
            raise ValueError('Supply an SSH host alias or user@host')
        if not 0 <= int(port) <= 65535:
            raise ValueError('Port must be between 0 and 65535')
        if int(port)==0:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0))
                port=sock.getsockname()[1]
        self.port=int(port)
        self.url=f'http://127.0.0.1:{self.port}/'
        self.session_id=uuid.uuid4().hex
        command=[remote_python,'-m','glyph'] if remote_python else ['glyph']
        command += ['serve',str(path),'--port',str(self.port),'--no-open',
                    '--session-id',self.session_id,'--stop-on-stdin-close']
        args=['ssh','-T','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30',
              '-L',f'127.0.0.1:{self.port}:127.0.0.1:{self.port}',host,shlex.join(command)]
        self._process=subprocess.Popen(args,stdin=subprocess.PIPE)
        self._closed=False
        try:
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                if self._process.poll() is not None:
                    raise RuntimeError('SSH viewer exited. Check the SSH host, remote Glyph installation and HTML path.')
                try:
                    with urlopen(Request(self.url,method='HEAD'),timeout=.5) as response:
                        if response.headers.get('X-Glyph-Session')==self.session_id:
                            break
                except OSError:
                    pass
                time.sleep(.1)
            else:
                raise TimeoutError('Timed out waiting for the SSH viewer to become available')
            print(f'Glyph viewer: {self.url}',flush=True)
            if open_browser:_open_browser(self.url)
        except BaseException:
            self.close()
            raise

    def close(self):
        import subprocess
        if self._closed:return
        self._closed=True
        if self._process.stdin:
            try:self._process.stdin.close()
            except OSError:pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)

    def wait(self):
        try:self._process.wait()
        except KeyboardInterrupt:pass
        finally:self.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def serve_over_ssh(host,path,*,port=0,open_browser=True,remote_python=None,timeout=60):
    return RemoteReportServer(host,path,port=port,open_browser=open_browser,
                              remote_python=remote_python,timeout=timeout)
