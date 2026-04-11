"""VeriMem is a library first — no interactive CLI (only this usage hint)."""


def main() -> None:
    print("VeriMem — import the Memory API in Python:")
    print("  from verimem import Memory")
    print("  mem = Memory()")
    print(
        "  mem.remember('...'); mem.recall('...')  # default mode: hybrid; or mode='raw' / 'rerank' / 'hybrid_rerank'"
    )
    print("Docs: https://github.com/itachi-hue/verimem")
