from pathlib import Path


def test_readme_has_install_and_test_sections():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "## 3) Instalação" in readme
    assert "## 7) Qualidade e CI" in readme
