"""quantize_yolox.py'nin alt kume secimi icin testler.

Bastan alan bir secim temsili degildi: birlestirilmis sette image_id'ler
dosya adina gore atandigi icin ilk N goruntu tek kaynaktan gelir ve
`mendeley` val'inin %85'i bos oldugundan AP=0 uretiyordu.
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_quantize_module():
    """cv2/torch kurulu olmayabilir; modulu yalnizca fonksiyon icin yukleme.

    quantize_yolox.py ust duzeyde cv2 ve torch import ediyor. Test ortaminda
    bunlar olmayabilecegi icin fonksiyonu kaynaktan ayikliyoruz.
    """
    source = (ROOT / "quantize" / "quantize_yolox.py").read_text(encoding="utf-8")
    start = source.index("def select_subset(")
    end = source.index("def evaluate(", start)
    namespace = {}
    exec(compile(source[start:end], "select_subset", "exec"), namespace)
    return namespace["select_subset"]


select_subset = load_quantize_module()


class SubsetSelectionTests(unittest.TestCase):
    def test_no_subset_returns_everything(self):
        ids = list(range(1, 101))
        self.assertEqual(select_subset(ids, None), ids)
        self.assertEqual(select_subset(ids, 0), ids)
        self.assertEqual(select_subset(ids, 500), ids)

    def test_requested_count_is_honoured(self):
        ids = list(range(1, 4484))  # gercek val boyutu
        for n in (1, 50, 300, 1000):
            self.assertEqual(len(select_subset(ids, n)), n)

    def test_ids_are_unique_and_ordered(self):
        ids = list(range(1, 4484))
        picked = select_subset(ids, 300)
        self.assertEqual(len(set(picked)), len(picked))
        self.assertEqual(picked, sorted(picked))

    def test_subset_spans_all_sources(self):
        """Asil mesele bu: secim tum kaynaklara yayilmali.

        Gercek val dagilimini taklit ediyoruz -- id'ler dosya adina gore
        atandigi icin kaynaklar bitisik bloklar halinde.
        """
        blocks = [("mendeley", 1516), ("milrec", 714),
                  ("vesselimg", 1705), ("visdrone", 548)]
        source_of = {}
        next_id = 1
        for name, count in blocks:
            for _ in range(count):
                source_of[next_id] = name
                next_id += 1
        ids = sorted(source_of)
        self.assertEqual(len(ids), 4483)

        picked = select_subset(ids, 50)
        covered = {source_of[i] for i in picked}
        self.assertEqual(covered, {"mendeley", "milrec", "vesselimg", "visdrone"},
                         "alt kume dort kaynagi da kapsamali")

        head = ids[:50]
        self.assertEqual({source_of[i] for i in head}, {"mendeley"},
                         "bastan alma tek kaynakta kalir -- duzeltilen davranis")


if __name__ == "__main__":
    unittest.main()
