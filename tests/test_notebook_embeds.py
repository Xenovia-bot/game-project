import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "training" / "kaggle_visdrone_yolox.ipynb"

EMBEDS = {
    3: ROOT / "tools" / "visdrone2coco.py",
    6: ROOT / "training" / "visdrone_eval.py",
    7: ROOT / "training" / "exps" / "yolox_nano_visdrone.py",
}


def notebook_writefile_body(index):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][index]["source"])
    first, _, body = source.partition("\n")
    if not first.startswith("%%writefile "):
        raise AssertionError("Beklenen writefile hucresi degil: %d" % index)
    return body


def strip_notebook_note(body, path):
    rel = path.relative_to(ROOT).as_posix()
    note = "\n\nBu dosya, projedeki %s ile ayni icerige sahiptir.\n" % rel
    return body.replace(note, "\n")


class NotebookEmbedTests(unittest.TestCase):
    def test_writefile_cells_match_project_sources(self):
        for index, path in EMBEDS.items():
            body = notebook_writefile_body(index)
            if path.name in {"visdrone2coco.py", "yolox_nano_visdrone.py"}:
                body = strip_notebook_note(body, path)
            self.assertEqual(
                body.replace("\r\n", "\n"),
                path.read_text(encoding="utf-8").replace("\r\n", "\n"),
                msg="Notebook cell %d drifted from %s" % (index, path),
            )

    def test_embedded_converter_preserves_ignore_semantics(self):
        namespace = {}
        exec(
            compile(notebook_writefile_body(3), "visdrone2coco.py", "exec"),
            namespace,
        )
        convert = namespace["convert"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, annotations = root / "images", root / "annotations"
            images.mkdir()
            annotations.mkdir()
            Image.new("RGB", (100, 100)).save(images / "one.jpg")
            (annotations / "one.txt").write_text(
                "0,0,40,40,0,0,0,0\n"
                "5,5,10,10,1,1,0,0\n"
                "50,50,10,10,1,1,0,0\n"
            )
            output = root / "out.json"
            convert(images, annotations, output)
            data = json.loads(output.read_text())
            self.assertEqual(data["images"][0]["ignore_regions"], [[0, 0, 40, 40]])
            self.assertEqual(len(data["annotations"]), 1)
            self.assertEqual(data["annotations"][0]["bbox"], [50, 50, 10, 10])


if __name__ == "__main__":
    unittest.main()
