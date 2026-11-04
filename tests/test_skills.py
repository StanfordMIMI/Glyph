import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from glyph.skill_cli import _catalog, main


def test_lists_bundled_skills(capsys):
    assert main(['skills', 'list', '--json']) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert {entry['name'] for entry in catalog} == {'glyph-discovery', 'glyph-literature', 'glyph-html', 'glyph-report'}
    assert all(entry['description'] and (Path(entry['path']) / 'SKILL.md').is_file()
               for entry in catalog)


def test_can_read_a_skill_without_installing_it(capsys):
    entry = next(entry for entry in _catalog() if entry['name'] == 'glyph-literature')
    main(['skills', 'show', entry['name']])
    assert capsys.readouterr().out.strip() == (Path(entry['path']) / 'SKILL.md').read_text().strip()


def test_project_install_copies_self_contained_skills_for_both_agents(tmp_path):
    main(['skills', 'install', '--agent', 'both', '--project', str(tmp_path)])
    for directory in ['.agents', '.claude']:
        for entry in _catalog():
            installed = tmp_path / directory / 'skills' / entry['name']
            assert (installed / 'SKILL.md').read_bytes() == (Path(entry['path']) / 'SKILL.md').read_bytes()
        assert (tmp_path / directory / 'skills/glyph-literature/references/comparison-workflow.md').exists()


def test_user_scope_and_single_skill_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    main(['skills', 'install', '--agent', 'codex', '--scope', 'user', '--name', 'glyph-html'])
    assert (tmp_path / '.agents/skills/glyph-html/SKILL.md').exists()
    assert not (tmp_path / '.agents/skills/glyph-discovery').exists()
    assert not (tmp_path / '.claude').exists()


def test_existing_skills_are_not_overwritten_or_partially_installed(tmp_path):
    target = tmp_path / '.claude/skills/glyph-html'
    target.mkdir(parents=True)
    (target / 'SKILL.md').write_text('Local customization')
    with pytest.raises(SystemExit) as error:
        main(['skills', 'install', '--agent', 'both', '--project', str(tmp_path)])
    assert error.value.code == 2
    assert (target / 'SKILL.md').read_text() == 'Local customization'
    assert not (tmp_path / '.agents').exists()
    main(['skills', 'install', '--agent', 'claude', '--project', str(tmp_path),
          '--name', 'glyph-html', '--force'])
    assert (target / 'SKILL.md').read_text() != 'Local customization'


def test_unknown_skill_does_not_create_directories(tmp_path):
    with pytest.raises(SystemExit):
        main(['skills', 'install', '--agent', 'codex', '--project', str(tmp_path), '--name', '../bad'])
    assert list(tmp_path.iterdir()) == []


def test_skill_discovery_does_not_import_training_dependencies():
    result = subprocess.run([sys.executable, '-c',
        "import sys; from glyph.skill_cli import main; main(['skills','list']); "
        "assert 'torch' not in sys.modules; assert 'pytorch_lightning' not in sys.modules"],
        env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_installed_harness_catalog_is_discoverable_without_torch(capsys):
    assert main(['harness','list'])==0
    output=capsys.readouterr().out
    assert 'literature-search' in output and 'literature-study' in output and 'interpret' in output and 'report' in output
    result=subprocess.run([sys.executable,'-B','-c',
        "import sys; from glyph.skill_cli import main; main(['harness','list']); assert 'torch' not in sys.modules"],
        env=os.environ.copy(),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
