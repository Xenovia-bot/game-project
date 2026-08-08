/*
 * tracker.hpp - IoU tabanli cok nesneli takip (ByteTrack'in iki asamali
 * eslestirme fikriyle), Kalman filtresi ve Macar algoritmasi olmadan.
 *
 * Neden takip?  Kare bazinda recall dusuk olsa bile (kucuk nesnelerde
 * kacinilmaz), bir hedef 30 karenin 18'inde gorulurse takip katmani izi
 * kesintisiz tutar; iz bazinda recall belirgin sekilde yukselir. Ayrica
 * "N karede gorunmeli" kurali, dusuk guven esiginde (0.15) uretilen yanlis
 * pozitifleri zamansal olarak eler. Bu ikisi birlikte, esigi dusurup
 * recall kazanmayi mumkun kilar.
 *
 * ByteTrack'ten alinan tek fikir su: yuksek skorlu tespitlerle eslestirmeyi
 * bitirdikten sonra, eslesmemis izleri **dusuk skorlu** tespitlerle
 * kurtarmayi dene. Kismen kapanan nesneler boylece kaybedilmez. Dusuk skorlu
 * tespitlerden yeni iz **acilmaz** - yoksa gurultu iz uretir.
 *
 * Hareket modeli: sabit hiz. Kalman yerine son yer degistirmenin ustel
 * hareketli ortalamasi kullanilir; hizli giden araclarda ardisik karelerde
 * IoU'nun esigin altina dusmesini onler ve ~10 satir tutar.
 *
 * Bilerek bagimsiz tutuldu: VART, OpenCV veya XIR icermez, boylece kart
 * disinda derlenip test edilebilir (bkz. deploy/tests/test_tracker.cpp).
 */
#ifndef YOLOX_VISDRONE_TRACKER_HPP_
#define YOLOX_VISDRONE_TRACKER_HPP_

#include <algorithm>
#include <cstddef>
#include <vector>

namespace tracking {

struct Box {
  float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;

  float width() const { return x2 - x1; }
  float height() const { return y2 - y1; }
  float cx() const { return (x1 + x2) * 0.5f; }
  float cy() const { return (y1 + y2) * 0.5f; }
  float area() const {
    const float w = width(), h = height();
    return (w > 0.0f && h > 0.0f) ? w * h : 0.0f;
  }
};

inline float iou(const Box& a, const Box& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float inter =
      std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
  const float uni = a.area() + b.area() - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

//! Tek bir kareye ait tespit.
struct Observation {
  Box box;
  float score = 0.0f;
  int cls = 0;
};

struct Track {
  int id = 0;
  Box box;
  int cls = 0;
  float score = 0.0f;
  float vx = 0.0f, vy = 0.0f;  //!< kare basina yer degistirme tahmini
  int hits = 0;                //!< toplam eslesme sayisi
  int age = 0;                 //!< son eslesmeden bu yana gecen kare sayisi
  bool confirmed = false;

  //! Bu karede tespitle eslesmis ve onaylanmis izler ciziler/loglanir.
  bool visible() const { return confirmed && age == 0; }
};

struct Config {
  //! Birinci asama esigi: bunun ustundeki tespitler yeni iz acabilir.
  float high_thr = 0.50f;
  //! Eslestirme icin gereken en dusuk IoU.
  float iou_thr = 0.30f;
  //! Bu kadar kare boyunca eslesmeyen iz silinir.
  int max_age = 30;
  //! Bu kadar eslesmeden sonra iz onaylanir (yanlis pozitif filtresi).
  int n_init = 3;
  //! Hiz tahmininin ustel hareketli ortalama katsayisi.
  float velocity_smoothing = 0.5f;
};

class Tracker {
 public:
  Tracker() = default;
  explicit Tracker(const Config& cfg) : cfg_(cfg) {}

  const Config& config() const { return cfg_; }
  const std::vector<Track>& tracks() const { return tracks_; }

  //! Bir kareyi isler ve guncel iz listesini dondurur.
  const std::vector<Track>& update(const std::vector<Observation>& dets) {
    predict();

    std::vector<char> track_taken(tracks_.size(), 0);
    std::vector<char> det_taken(dets.size(), 0);

    // 1. asama: yuksek skorlu tespitlerle eslestir.
    associate(dets, det_taken, track_taken, /*use_high=*/true);
    // 2. asama (ByteTrack): kalan izleri dusuk skorlu tespitlerle kurtar.
    associate(dets, det_taken, track_taken, /*use_high=*/false);

    // Eslesmeyen izler yaslanir; siniri asanlar silinir.
    for (std::size_t i = 0; i < tracks_.size(); ++i) {
      if (!track_taken[i]) ++tracks_[i].age;
    }
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [this](const Track& t) { return t.age > cfg_.max_age; }),
        tracks_.end());

    // Eslesmeyen **yuksek** skorlu tespitler yeni iz acar.
    for (std::size_t d = 0; d < dets.size(); ++d) {
      if (det_taken[d] || dets[d].score < cfg_.high_thr) continue;
      Track t;
      t.id = next_id_++;
      t.box = dets[d].box;
      t.cls = dets[d].cls;
      t.score = dets[d].score;
      t.hits = 1;
      t.age = 0;
      t.confirmed = (cfg_.n_init <= 1);
      tracks_.push_back(t);
    }
    return tracks_;
  }

  void reset() {
    tracks_.clear();
    next_id_ = 1;
  }

 private:
  struct Candidate {
    float score;  // IoU
    std::size_t track_index;
    std::size_t det_index;
  };

  //! Sabit hiz varsayimiyla her izin kutusunu bir kare ileri tasir.
  void predict() {
    for (Track& t : tracks_) {
      t.box.x1 += t.vx;
      t.box.x2 += t.vx;
      t.box.y1 += t.vy;
      t.box.y2 += t.vy;
    }
  }

  void associate(const std::vector<Observation>& dets,
                 std::vector<char>& det_taken,
                 std::vector<char>& track_taken, bool use_high) {
    std::vector<Candidate> candidates;
    for (std::size_t i = 0; i < tracks_.size(); ++i) {
      if (track_taken[i]) continue;
      for (std::size_t d = 0; d < dets.size(); ++d) {
        if (det_taken[d]) continue;
        const bool is_high = dets[d].score >= cfg_.high_thr;
        if (is_high != use_high) continue;
        if (dets[d].cls != tracks_[i].cls) continue;  // sinif bazli eslestirme
        const float overlap = iou(tracks_[i].box, dets[d].box);
        if (overlap < cfg_.iou_thr) continue;
        candidates.push_back({overlap, i, d});
      }
    }

    // Acgozlu eslestirme: en yuksek IoU once. Beraberlikler indekse gore
    // cozulur, boylece sonuc calistirmadan calistirmaya degismez.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                if (a.score != b.score) return a.score > b.score;
                if (a.track_index != b.track_index)
                  return a.track_index < b.track_index;
                return a.det_index < b.det_index;
              });

    for (const Candidate& c : candidates) {
      if (track_taken[c.track_index] || det_taken[c.det_index]) continue;
      track_taken[c.track_index] = 1;
      det_taken[c.det_index] = 1;
      apply(tracks_[c.track_index], dets[c.det_index]);
    }
  }

  void apply(Track& t, const Observation& det) {
    // Hiz: tahmin edilen konumdan olculen konuma sapma, EMA ile yumusatilir.
    const float dx = det.box.cx() - t.box.cx();
    const float dy = det.box.cy() - t.box.cy();
    const float a = cfg_.velocity_smoothing;
    t.vx = (1.0f - a) * t.vx + a * (t.vx + dx);
    t.vy = (1.0f - a) * t.vy + a * (t.vy + dy);

    t.box = det.box;
    t.score = det.score;
    t.age = 0;
    ++t.hits;
    if (t.hits >= cfg_.n_init) t.confirmed = true;
  }

  Config cfg_;
  std::vector<Track> tracks_;
  int next_id_ = 1;
};

}  // namespace tracking

#endif  // YOLOX_VISDRONE_TRACKER_HPP_
