from pathlib import Path
from shutil import copytree, ignore_patterns

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithSkills(build_py):
    def run(self):
        super().run()
        copytree('skills', Path(self.build_lib) / 'glyph' / 'agent_skills',
                 dirs_exist_ok=True, ignore=ignore_patterns('__pycache__', '*.pyc'))


# Keep the SVG in GitHub’s README; use a versioned public PNG for PyPI.
PYPI_IMAGE_URL = (
    "https://raw.githubusercontent.com/StanfordMIMI/Glyph/v0.2.2/Glyph.png"
)
readme = (Path(__file__).parent / 'README.md').read_text(encoding='utf-8')
pypi_readme = readme.replace('src="Glyph.svg"', f'src="{PYPI_IMAGE_URL}"', 1).replace(
    'src="https://raw.githubusercontent.com/StanfordMIMI/Glyph/main/Glyph.svg"',
    f'src="{PYPI_IMAGE_URL}"', 1,
)

setup(
    cmdclass={'build_py': BuildWithSkills},
    long_description=pypi_readme,
    long_description_content_type='text/markdown',
)
