// tracker.hpp icin bagimsiz testler. VART/OpenCV gerektirmez, bu yuzden
// kart disinda da derlenip calistirilabilir:
//
//   g++ -std=c++17 -O2 -I deploy/src deploy/tests/test_tracker.cpp -o tracker_test
//   ./tracker_test
//
// `tests/test_cpp_tracker.py` bunu otomatik derleyip calistirir.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tracker.hpp"

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

tracking::Observation obs(float x, float y, float w, float h, float score,
                          int cls = 0) {
  tracking::Observation o;
  o.box = {x, y, x + w, y + h};
  o.score = score;
  o.cls = cls;
  return o;
}

const tracking::Track* find_track(const std::vector<tracking::Track>& tracks,
                                  int id) {
  for (const auto& t : tracks) {
    if (t.id == id) return &t;
  }
  return nullptr;
}

int visible_count(const std::vector<tracking::Track>& tracks) {
  int n = 0;
  for (const auto& t : tracks) {
    if (t.visible()) ++n;
  }
  return n;
}

// --------------------------------------------------------------------------

void test_track_is_confirmed_after_n_init() {
  std::printf("test_track_is_confirmed_after_n_init\n");
  tracking::Config cfg;
  cfg.n_init = 3;
  tracking::Tracker tracker(cfg);

  for (int frame = 0; frame < 2; ++frame) {
    const auto& tracks = tracker.update({obs(100, 100, 40, 40, 0.9f)});
    check(tracks.size() == 1, "iz olusmali");
    check(!tracks[0].confirmed, "n_init'ten once onaylanmamali");
    check(visible_count(tracks) == 0, "onaylanmadan gorunur olmamali");
  }
  const auto& tracks = tracker.update({obs(100, 100, 40, 40, 0.9f)});
  check(tracks[0].confirmed, "3. eslesmede onaylanmali");
  check(visible_count(tracks) == 1, "onaylanan iz gorunur olmali");
}

void test_moving_object_keeps_one_id() {
  std::printf("test_moving_object_keeps_one_id\n");
  tracking::Tracker tracker;
  int first_id = -1;
  // Kare basina 15 px saga giden bir arac; 40 px'lik kutuda bu, hareket
  // modeli olmasa IoU'yu esige yaklastiracak bir hiz.
  for (int frame = 0; frame < 10; ++frame) {
    const auto& tracks =
        tracker.update({obs(100.0f + 15 * frame, 100, 40, 40, 0.9f)});
    check(tracks.size() == 1, "tek nesne tek iz uretmeli");
    if (first_id < 0) first_id = tracks[0].id;
    check(tracks[0].id == first_id, "id degismemeli");
  }
}

void test_missed_frames_do_not_break_track() {
  std::printf("test_missed_frames_do_not_break_track\n");
  tracking::Config cfg;
  cfg.max_age = 5;
  tracking::Tracker tracker(cfg);

  for (int frame = 0; frame < 4; ++frame) {
    tracker.update({obs(100, 100, 40, 40, 0.9f)});
  }
  const int id = tracker.tracks()[0].id;

  for (int frame = 0; frame < 3; ++frame) {  // 3 kare hic tespit yok
    const auto& tracks = tracker.update({});
    check(tracks.size() == 1, "iz max_age dolmadan silinmemeli");
    check(!tracks[0].visible(), "eslesmeyen iz gorunur sayilmamali");
  }
  const auto& tracks = tracker.update({obs(100, 100, 40, 40, 0.9f)});
  check(tracks.size() == 1, "iz kurtarilmali");
  check(tracks[0].id == id, "kurtarilan iz ayni id'yi tasimali");
  check(tracks[0].visible(), "yeniden eslesen iz gorunur olmali");
}

void test_track_expires_after_max_age() {
  std::printf("test_track_expires_after_max_age\n");
  tracking::Config cfg;
  cfg.max_age = 3;
  tracking::Tracker tracker(cfg);
  for (int frame = 0; frame < 4; ++frame) {
    tracker.update({obs(100, 100, 40, 40, 0.9f)});
  }
  for (int frame = 0; frame < 4; ++frame) {
    tracker.update({});
  }
  check(tracker.tracks().empty(), "max_age asilinca iz silinmeli");
}

void test_low_score_detection_rescues_track() {
  std::printf("test_low_score_detection_rescues_track\n");
  tracking::Config cfg;
  cfg.high_thr = 0.50f;
  tracking::Tracker tracker(cfg);
  for (int frame = 0; frame < 4; ++frame) {
    tracker.update({obs(100, 100, 40, 40, 0.9f)});
  }
  const int id = tracker.tracks()[0].id;

  // ByteTrack 2. asama: 0.20 skorlu tespit yeni iz acamaz ama mevcut izi
  // kurtarabilmelidir (kismi kapanma senaryosu).
  const auto& tracks = tracker.update({obs(100, 100, 40, 40, 0.20f)});
  check(tracks.size() == 1, "dusuk skorlu tespit yeni iz acmamali");
  check(tracks[0].id == id, "mevcut iz kurtarilmali");
  check(tracks[0].age == 0, "kurtarilan izin yasi sifirlanmali");
}

void test_low_score_detection_alone_creates_no_track() {
  std::printf("test_low_score_detection_alone_creates_no_track\n");
  tracking::Tracker tracker;
  for (int frame = 0; frame < 10; ++frame) {
    const auto& tracks = tracker.update({obs(100, 100, 40, 40, 0.20f)});
    check(tracks.empty(), "yalniz dusuk skorlu tespitler iz uretmemeli");
  }
}

void test_two_objects_do_not_swap_ids() {
  std::printf("test_two_objects_do_not_swap_ids\n");
  tracking::Tracker tracker;
  int left_id = -1, right_id = -1;
  // Birbirine dogru yaklasan ama carpismayan iki nesne.
  for (int frame = 0; frame < 8; ++frame) {
    const float left = 100.0f + 5 * frame;
    const float right = 400.0f - 5 * frame;
    const auto& tracks =
        tracker.update({obs(left, 100, 40, 40, 0.9f),
                        obs(right, 100, 40, 40, 0.9f)});
    check(tracks.size() == 2, "iki nesne iki iz uretmeli");
    if (left_id < 0) {
      left_id = tracks[0].box.x1 < tracks[1].box.x1 ? tracks[0].id : tracks[1].id;
      right_id = tracks[0].box.x1 < tracks[1].box.x1 ? tracks[1].id : tracks[0].id;
    }
    const tracking::Track* l = find_track(tracks, left_id);
    const tracking::Track* r = find_track(tracks, right_id);
    check(l != nullptr && r != nullptr, "her iki iz de yasamali");
    if (l && r) check(l->box.x1 < r->box.x1, "id'ler yer degistirmemeli");
  }
}

void test_classes_never_match_each_other() {
  std::printf("test_classes_never_match_each_other\n");
  tracking::Tracker tracker;
  for (int frame = 0; frame < 4; ++frame) {
    tracker.update({obs(100, 100, 40, 40, 0.9f, /*cls=*/0)});
  }
  const int person_id = tracker.tracks()[0].id;

  // Ayni konumda farkli sinif: mevcut ize eslesmemeli, yeni iz acmali.
  const auto& tracks = tracker.update({obs(100, 100, 40, 40, 0.9f, /*cls=*/1)});
  check(tracks.size() == 2, "farkli sinif ayri iz olmali");
  const tracking::Track* person = find_track(tracks, person_id);
  check(person != nullptr && person->age == 1,
        "person izi eslesmeden yaslanmali");
}

void test_reset_clears_state() {
  std::printf("test_reset_clears_state\n");
  tracking::Tracker tracker;
  tracker.update({obs(100, 100, 40, 40, 0.9f)});
  tracker.reset();
  check(tracker.tracks().empty(), "reset izleri temizlemeli");
  const auto& tracks = tracker.update({obs(100, 100, 40, 40, 0.9f)});
  check(tracks[0].id == 1, "reset id sayacini sifirlamali");
}

void test_iou_basics() {
  std::printf("test_iou_basics\n");
  const tracking::Box a{0, 0, 10, 10};
  check(tracking::iou(a, a) > 0.999f, "ayni kutu IoU=1");
  const tracking::Box far{100, 100, 110, 110};
  check(tracking::iou(a, far) == 0.0f, "ayrik kutular IoU=0");
  const tracking::Box degenerate{0, 0, 0, 0};
  check(tracking::iou(a, degenerate) == 0.0f, "bozuk kutu sifir donmeli");
}

}  // namespace

int main() {
  test_iou_basics();
  test_track_is_confirmed_after_n_init();
  test_moving_object_keeps_one_id();
  test_missed_frames_do_not_break_track();
  test_track_expires_after_max_age();
  test_low_score_detection_rescues_track();
  test_low_score_detection_alone_creates_no_track();
  test_two_objects_do_not_swap_ids();
  test_classes_never_match_each_other();
  test_reset_clears_state();

  if (g_failures == 0) {
    std::printf("\nTUM TRACKER TESTLERI GECTI\n");
    return 0;
  }
  std::printf("\n%d test basarisiz\n", g_failures);
  return 1;
}
