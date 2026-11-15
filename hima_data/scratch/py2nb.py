import json, re, sys
src = open(sys.argv[1]).read()
cells = []
for chunk in re.split(r"(?m)^# %%", src)[1:]:
    head, _, body = chunk.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in head:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body.splitlines(True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w"), indent=1)
print(len(cells), "cells ->", sys.argv[2])
