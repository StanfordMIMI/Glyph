"""Discover and install the agent skills bundled with Glyph."""

import argparse
import json
import re
import shutil
from pathlib import Path


def _skill_root():
    bundled = Path(__file__).parent / 'agent_skills'
    return bundled if bundled.is_dir() else Path(__file__).resolve().parents[1] / 'skills'


def _catalog():
    entries = []
    for manifest in sorted(_skill_root().glob('*/SKILL.md')):
        content = manifest.read_text(encoding='utf-8')
        frontmatter = content.split('---', 2)[1]
        metadata = dict(re.findall(r'^(name|description):[ \t]*(.+)$', frontmatter, re.MULTILINE))
        entries.append(dict(name=metadata['name'], description=metadata['description'],
                            path=str(manifest.parent)))
    return entries


def _install(entries, agent, scope, project, force):
    root = Path.home() if scope == 'user' else Path(project).resolve()
    agents = ['codex', 'claude'] if agent == 'both' else [agent]
    destinations = []
    for selected_agent in agents:
        directory = '.agents' if selected_agent == 'codex' else '.claude'
        for entry in entries:
            destinations.append((Path(entry['path']), root / directory / 'skills' / entry['name']))
    for source, destination in destinations:
        if destination.is_symlink():
            raise ValueError(f'Refusing to replace a symlink: {destination}')
        if destination.exists() and not force:
            raise ValueError(f'Skill already exists: {destination}. Use --force to replace it.')
    for source, destination in destinations:
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        print(f'Installed {destination}')
    print('Skills are ready for your next agent turn. Restart the agent if they are not listed.')



HARNESSES={
    'literature-search':('glyph-literature','discover.py',[]),
    'literature-freeze':('glyph-literature','workflow.py',['freeze']),
    'literature-study':('glyph-literature','workflow.py',['study']),
    'literature-compare':('glyph-literature','compare.py',[]),
    'interpret':('glyph-discovery','interpret.py',[]),
    'report':('glyph-report','build_report.py',[]),
}


def _harness(argv):
    if not argv or argv[0] in ['list','--help','-h']:
        print('Bundled agent harnesses:')
        for name in HARNESSES:print('  '+name)
        print('Run: glyph harness NAME --help')
        return 0
    name=argv[0]
    if name not in HARNESSES:raise ValueError('Unknown harness; run glyph harness list')
    import subprocess
    import sys
    skill,script,prefix=HARNESSES[name]
    path=_skill_root()/skill/'scripts'/script
    return subprocess.call([sys.executable,'-B',str(path),*prefix,*argv[1:]])


def main(argv=None):
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv[:1]==['harness']:
        try:return _harness(argv[1:])
        except ValueError as exc:raise SystemExit(str(exc))
    parser = argparse.ArgumentParser(
        prog='glyph', description='Glyph viewer and bundled agent skills.',
        epilog='After pip install: glyph skills list; glyph skills install --agent codex',
    )
    commands = parser.add_subparsers(dest='command')
    commands.add_parser('harness',help='Run bundled agent literature, interpretation and report scripts')
    serve = commands.add_parser('serve', help='Host a generated HTML viewer and open a local browser tab')
    serve.add_argument('path', help='HTML file, or directory of generated HTML maps')
    serve.add_argument('--port', type=int, default=0, help='Loopback port; default chooses a free port')
    serve.add_argument('--no-open', action='store_true', help='Print the URL without opening a browser')
    serve.add_argument('--ssh', metavar='HOST', help='Run on your laptop: host remotely through SSH and open locally')
    serve.add_argument('--remote-python', help='Remote Python executable with Glyph installed; default invokes glyph')
    serve.add_argument('--timeout', type=float, default=60, help='Seconds to wait for remote viewer startup')
    serve.add_argument('--session-id', help=argparse.SUPPRESS)
    serve.add_argument('--stop-on-stdin-close', action='store_true', help=argparse.SUPPRESS)
    skills = commands.add_parser('skills' , help='List, read or install bundled agent skills')
    actions = skills.add_subparsers(dest='action')
    list_parser = actions.add_parser('list', help='Show the skills included in this installation')
    list_parser.add_argument('--json', action='store_true', dest='as_json')
    show = actions.add_parser('show', help='Print a skill without installing it')
    show.add_argument('name')
    install = actions.add_parser('install', help='Install skills for Claude Code or Codex')
    install.add_argument('--agent', choices=['codex', 'claude', 'both'], required=True)
    install.add_argument('--scope', choices=['project', 'user'], default='project')
    install.add_argument('--project', type=Path, default=Path.cwd())
    install.add_argument('--name', action='append', help='Install only a named skill; repeat as needed')
    install.add_argument('--force', action='store_true', help='Replace existing skills with these names')
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == 'serve':
        from glyph.visualization.hosting import serve_html, serve_over_ssh
        try:
            if args.ssh:
                server = serve_over_ssh(args.ssh,args.path,port=args.port,open_browser=not args.no_open,
                                       remote_python=args.remote_python,timeout=args.timeout)
            else:
                server = serve_html(args.path,port=args.port,open_browser=not args.no_open,session_id=args.session_id)
                if args.stop_on_stdin_close:
                    import threading
                    import sys
                    def stop_on_eof():
                        sys.stdin.buffer.read()
                        server.close()
                    threading.Thread(target=stop_on_eof,daemon=True).start()
            server.wait()
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        return 0
    if args.action is None:
        skills.print_help()
        return 0
    entries = _catalog()
    if args.action == 'list':
        if args.as_json:
            print(json.dumps(entries, indent=2))
        else:
            for entry in entries:
                print(f"{entry['name']}: {entry['description']}")
            print('\nRead: glyph skills show NAME')
            print('Install: glyph skills install --agent codex   # or claude / both')
        return 0
    names = [args.name] if args.action == 'show' else args.name
    unknown = set(names or []) - {entry['name'] for entry in entries}
    if unknown:
        parser.error(f'Unknown skills: {", ".join(sorted(unknown))}')
    selected = [entry for entry in entries if names is None or entry['name'] in names]
    if args.action == 'show':
        print((Path(selected[0]['path']) / 'SKILL.md').read_text(encoding='utf-8'))
    else:
        try:
            _install(selected, args.agent, args.scope, args.project, args.force)
        except ValueError as exc:
            parser.error(str(exc))
    return 0
