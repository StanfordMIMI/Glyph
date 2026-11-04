from pathlib import Path
from shutil import copytree, ignore_patterns

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithSkills(build_py):
    def run(self):
        super().run()
        copytree('skills', Path(self.build_lib) / 'glyph' / 'agent_skills',
                 dirs_exist_ok=True, ignore=ignore_patterns('__pycache__', '*.pyc'))


setup(cmdclass={'build_py': BuildWithSkills})
