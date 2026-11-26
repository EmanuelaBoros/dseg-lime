import sys, nbformat
from nbclient import NotebookClient
src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name="hier-attr", resources={"metadata": {"path": "notebooks"}})
try:
    client.execute()
finally:
    nbformat.write(nb, dst)
print("written", dst)
