from pathlib import Path


def test_readme_has_setup_section():
    readme = Path('README.md').read_text(encoding='utf-8')
    assert '## Setup' in readme
    assert '## Executar testes' in readme
