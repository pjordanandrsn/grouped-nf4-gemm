"""Extract the CPU-quickstart fenced blocks from the README and execute them.

The README promises three copy-pasteable CPU demos with real call shapes; this
test is the structural guarantee that they never drift from the API. It reads
the ```python blocks between the <!-- CPU-QUICKSTART-START/END --> markers,
execs each, captures stdout, and asserts every printed boolean is True and
every printed rel-err is a plausible quantization error. A README example that
goes stale now fails the build — the same invariant class as the wheel-surface
smoke.
"""
import io
import os
import re
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, os.pardir, "README.md")


def _cpu_blocks():
    text = open(README, encoding="utf-8").read()
    m = re.search(r"<!-- CPU-QUICKSTART-START -->(.*?)<!-- CPU-QUICKSTART-END -->", text, re.S)
    assert m, "CPU-QUICKSTART markers not found in README.md"
    blocks = re.findall(r"```python\n(.*?)```", m.group(1), re.S)
    assert len(blocks) == 3, f"expected 3 CPU demo blocks, found {len(blocks)}"
    return blocks


def test_readme_cpu_blocks_execute_and_assert():
    for i, block in enumerate(_cpu_blocks()):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(block, f"<readme-cpu-block-{i}>", "exec"), {})
        out = buf.getvalue()
        assert out.strip(), f"block {i} printed nothing"
        for line in out.splitlines():
            if line.strip().endswith(("True", "False")):
                assert line.strip().endswith("True"), f"block {i}: {line!r} not True"
            if "rel-err:" in line:
                val = float(line.rsplit(":", 1)[1])
                assert 0.0 < val < 0.2, f"block {i}: rel-err {val} out of band"
