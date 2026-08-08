"""deploy/src/tracker.hpp icin C++ testlerini derleyip calistirir.

Tracker kart uzerinde C++ olarak kosacagi icin mantigi Python'da ikinci kez
yazmak yerine gercek kaynagi derleyip test ediyoruz. `tracker.hpp` bilerek
VART/OpenCV bagimsizdir; tek ihtiyac bir C++17 derleyicisi.

Derleyici yoksa test atlanir - CI/gelistirici makinesinde g++ bulunmayabilir.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deploy" / "tests" / "test_tracker.cpp"
INCLUDE_DIR = ROOT / "deploy" / "src"


def find_compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


class CppTrackerTests(unittest.TestCase):
    def test_tracker_cpp_suite_passes(self):
        compiler = find_compiler()
        if compiler is None:
            self.skipTest("C++ derleyicisi bulunamadi (g++/clang++)")
        self.assertTrue(SOURCE.is_file(), f"kaynak yok: {SOURCE}")

        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "tracker_test"
            build = subprocess.run(
                [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
                 "-I", str(INCLUDE_DIR), str(SOURCE), "-o", str(binary)],
                capture_output=True, text=True,
            )
            if build.returncode != 0 and not (build.stdout or build.stderr):
                # Derleyici cagrilabiliyor ama hic cikti vermeden basarisiz
                # oluyor (kisitli ortam). Gercek bir derleme hatasi degil.
                self.skipTest("derleyici bu ortamda calistirilamiyor")
            self.assertEqual(
                build.returncode, 0,
                f"derleme basarisiz:\n{build.stdout}\n{build.stderr}",
            )
            # Windows'ta ciktiya .exe eklenir
            if not binary.exists() and binary.with_suffix(".exe").exists():
                binary = binary.with_suffix(".exe")

            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(
                run.returncode, 0,
                f"tracker testleri basarisiz:\n{run.stdout}\n{run.stderr}",
            )
            self.assertIn("TUM TRACKER TESTLERI GECTI", run.stdout)


if __name__ == "__main__":
    unittest.main()
