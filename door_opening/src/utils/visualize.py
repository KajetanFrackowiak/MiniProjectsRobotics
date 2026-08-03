from pathlib import Path

from graphviz import Source


def make_diagram(diagram, name="iiwa_diagram"):
    src_dir = Path(__file__).resolve().parent.parent
    diagrams_dir = src_dir / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)

    dot = diagram.GetGraphvizString()

    Source(dot).render(
        filename=str(diagrams_dir / name),
        format="png",
        cleanup=True,
    )
