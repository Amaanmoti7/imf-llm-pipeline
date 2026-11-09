"""Assemble src_parts/*.py (percent-format cells) into the Colab notebook."""
import re, sys
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
src = "\n".join(p.read_text() for p in sorted((ROOT / "src_parts").glob("p*.py")))
cells = []
for block in re.split(r"^# %%", src, flags=re.MULTILINE):
    if not block.strip():
        continue
    if block.startswith(" [markdown]"):
        body = block[len(" [markdown]"):].strip("\n")
        md = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md.strip()))
    else:
        cells.append(nbformat.v4.new_code_cell(block.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True},
               "accelerator": "None"}
out = ROOT / "IMF_Dominica_LLM_Pipeline.ipynb"
nbformat.validate(nb)
nbformat.write(nb, out)
print(f"{out.name}: {len(cells)} cells ({sum(c.cell_type=='code' for c in cells)} code)")
