#!/usr/bin/env python3
"""
互換エントリポイント: ``map_cope_alt_dark`` と同じ HTML（``run_interactive_landscape_cope_vector_field``）で、
Φ だけ **密度由来（Φ = −log p̂）** にしたい場合のショートカット。

次と同等です（先頭に ``--phi-source density_kde`` を付与）::

  python -m pnode_patent_runner.run_interactive_landscape_cope_vector_field \\
    --phi-source density_kde \\
    ...他の引数はそのまま...

例::

  python -m pnode_patent_runner.run_interactive_landscape_mu_kde_html \\
    --load-checkpoint pnode_patent_runner/outputs/cope_landscape/unified_vgae.pt \\
    --year-range 2010 2020 \\
    --output pnode_patent_runner/outputs/offline_density_mu/map_mu_kde_alt_dark.html
"""
from __future__ import annotations

import runpy
import sys


def main() -> None:
    argv = [sys.argv[0], "--phi-source", "density_kde"] + sys.argv[1:]
    sys.argv = argv
    runpy.run_module("pnode_patent_runner.run_interactive_landscape_cope_vector_field", run_name="__main__")


if __name__ == "__main__":
    main()
