"""
Execute a notebook's code cells in one process and write the outputs back in.

nbconvert is unusable here: the venv's copy reaches into the conda environment
for nbclient/pyzmq, whose libzmq wants a GLIBCXX the system libstdc++ does not
provide. Running the cells directly needs none of that machinery, and a tutorial
is a linear script anyway -- which is the only case this handles.

    python run_notebook.py path/to/nb.ipynb [-o out.ipynb]

Exits non-zero on the first cell that raises, having recorded the traceback, so
the failure is visible in the notebook as well as on the console.
"""
import argparse
import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout


def run(path, out_path):
    nb = json.load(open(path, encoding="utf-8"))
    ns = {"__name__": "__main__"}
    n_code = failed = 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(cell["source"])
        buf = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                exec(compile(src, f"<cell {i}>", "exec"), ns)
            text = buf.getvalue()
            status = "ok"
        except BaseException:
            text = buf.getvalue() + "\n" + traceback.format_exc()
            status = "FAILED"
            failed = i

        cell["execution_count"] = n_code
        cell["outputs"] = ([{"output_type": "stream", "name": "stdout",
                             "text": text.splitlines(True)}] if text else [])

        head = next((l for l in src.splitlines() if l.strip()
                     and not l.startswith("#")), "")
        print(f"[cell {i:2d}] {status:6s}  {head[:64]}")
        if text.strip():
            for line in text.rstrip().splitlines():
                print(f"          | {line}")
        if status == "FAILED":
            break

    json.dump(nb, open(out_path, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    open(out_path, "a", encoding="utf-8").write("\n")
    print(f"\nwrote {out_path}")
    return failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    sys.exit(1 if run(a.notebook, a.out or a.notebook) else 0)
