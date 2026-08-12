/*
 * yolox_visdrone_demo
 * -------------------
 * KV260 uzerinde YOLOX-Nano video demosu.
 * Siniflar: land_vehicle, sea_vehicle.
 *
 * Akis: video karesi -> letterbox (114 dolgu, sol-ust) -> INT8 DPU girisi
 *       -> VART ile DPU kosumu -> YOLOX decode (grid + exp + sigmoid, LUT'lu)
 *       -> sinif bazli NMS -> orijinal koordinatlara donus
 *       -> IoU tabanli takip (iz kimligi + zamansal FP filtresi)
 *       -> MERKEZ NOKTASI: cx=(x1+x2)/2, cy=(y1+y2)/2
 *       -> cizim + cikti videosu + centers.csv
 *
 * Kullanim:
 *   ./yolox_visdrone_demo <model.xmodel> <video> [cikti.avi] [centers.csv]
 *                         [--conf 0.15] [--nms 0.45] [--max-frames N]
 *                         [--dump-first-frame DIR] [--no-track]
 *                         [--track-n-init 3] [--track-max-age 30]
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <xir/attrs/attrs.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

#include "tracker.hpp"

namespace {

// Saha tanimi. Sira, egitimdeki COCO kategori kimlikleriyle ayni olmali
// (tools/build_dataset.py: 1=land_vehicle, 2=sea_vehicle).
const char* kClassNames[] = {"land_vehicle", "sea_vehicle"};
constexpr int kNumClassNames = sizeof(kClassNames) / sizeof(kClassNames[0]);

struct Detection {
  float x1, y1, x2, y2;
  float score;
  int cls;
};

// Bir stride seviyesi: reg/obj/cls AYRI tensorlerdir.
//
// Neden ayri (2026-08-09'da olculdu): bu ucu modelde `torch.cat` ile tek
// tensore birlestirmek, Vitis AI'da hepsini AYNI fix_point'e zorluyor (concat
// donanimda yalnizca bellek duzenidir). obj logitleri -76'ya kadar inerken reg
// ofsetleri +-2.9 araliginda; ortak olcek reg'e 256 seviyenin ~16'sini
// birakiyor ve kutu merkezi stride 16'da 4 piksele kayiyordu.
//
// INT8 ciktilar yalnizca 256 farkli deger alabildiginden sigmoid/exp
// fonksiyonlari tensor basina LUT olarak bir kez hesaplanir. Her tensorun
// kendi fix_point'i oldugu icin LUT'lar da ayridir.
struct OutputLevel {
  vart::TensorBuffer* reg = nullptr;  // [1,H,W,4]  cx,cy,logw,logh (ham)
  vart::TensorBuffer* obj = nullptr;  // [1,H,W,1]  objectness logiti
  vart::TensorBuffer* cls = nullptr;  // [1,H,W,nc] sinif logitleri
  int height = 0;
  int width = 0;
  int stride = 0;
  int num_classes = 0;
  // dequantize: float = int8 * scale (scale = 2^-fix_point)
  float reg_scale = 1.0f;
  float obj_scale = 1.0f;
  float cls_scale = 1.0f;
  std::vector<float> reg_exp_lut;      // exp(v * reg_scale)
  std::vector<float> obj_sigmoid_lut;  // sigmoid(v * obj_scale)
  std::vector<float> cls_sigmoid_lut;  // sigmoid(v * cls_scale)
  int8_t obj_gate = 127;  // sigmoid(obj) >= conf kosulunun ham int8 karsiligi

  bool complete() const { return reg && obj && cls; }
};

// Bir cikti tensorunun ham int8 isaretcisini dogrulayarak alir.
const int8_t* tensor_data(vart::TensorBuffer* tb, size_t expected,
                          const char* what) {
  uint64_t addr = 0u;
  size_t size = 0u;
  std::tie(addr, size) = tb->data(std::vector<int>{0, 0, 0, 0});
  if (addr == 0u || size < expected) {
    std::cerr << "HATA: " << what << " tampon boyutu yetersiz\n";
    return nullptr;
  }
  return reinterpret_cast<const int8_t*>(addr);
}

float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float iou(const Detection& a, const Detection& b) {
  float xx1 = std::max(a.x1, b.x1);
  float yy1 = std::max(a.y1, b.y1);
  float xx2 = std::min(a.x2, b.x2);
  float yy2 = std::min(a.y2, b.y2);
  float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / std::max(area_a + area_b - inter, 1e-9f);
}

std::vector<Detection> nms_per_class(std::vector<Detection>& dets, float nms_thr) {
  std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.cls != b.cls) return a.cls < b.cls;
    if (a.x1 != b.x1) return a.x1 < b.x1;
    if (a.y1 != b.y1) return a.y1 < b.y1;
    if (a.x2 != b.x2) return a.x2 < b.x2;
    return a.y2 < b.y2;
  });
  std::vector<Detection> keep;
  std::vector<bool> removed(dets.size(), false);
  for (size_t i = 0; i < dets.size(); ++i) {
    if (removed[i]) continue;
    keep.push_back(dets[i]);
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (removed[j] || dets[j].cls != dets[i].cls) continue;
      if (iou(dets[i], dets[j]) > nms_thr) removed[j] = true;
    }
  }
  return keep;
}

double ms_since(const std::chrono::steady_clock::time_point& t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Kullanim: " << argv[0]
              << " <model.xmodel> <video> [cikti.avi] [centers.csv]"
                 " [--conf 0.15] [--nms 0.45] [--max-frames N]"
                 " [--dump-first-frame DIR] [--no-track] [--no-video]"
                 " [--track-n-init 3] [--track-max-age 30]\n";
    return 1;
  }
  std::string model_path = argv[1];
  std::string video_path = argv[2];
  std::string out_video_path = "output.avi";
  std::string csv_path = "centers.csv";
  // Recall odakli calisma noktasi: dusuk esikte tespit uret,
  // yanlis pozitifleri tracker'in "N karede gorunmeli" kurali elesin.
  float conf_thr = 0.15f;
  float nms_thr = 0.45f;
  long max_frames = -1;
  std::string dump_dir;
  bool use_tracker = true;
  bool draw_output = true;
  tracking::Config track_cfg;
  int positional = 0;
  try {
    for (int i = 3; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--conf" && i + 1 < argc) {
        conf_thr = std::stof(argv[++i]);
      } else if (a == "--nms" && i + 1 < argc) {
        nms_thr = std::stof(argv[++i]);
      } else if (a == "--max-frames" && i + 1 < argc) {
        max_frames = std::stol(argv[++i]);
      } else if (a == "--dump-first-frame" && i + 1 < argc) {
        dump_dir = argv[++i];
      } else if (a == "--no-track") {
        use_tracker = false;
      } else if (a == "--no-video") {
        // Urunun ciktisi centers.csv; aciklamali video yalnizca gorsel
        // kontrol icin. Yogun sahnede 300+ kutunun cizimi son-islemenin
        // buyuk kismini yiyor, dagitimda bu maliyet gereksiz.
        draw_output = false;
      } else if (a == "--track-n-init" && i + 1 < argc) {
        track_cfg.n_init = std::stoi(argv[++i]);
      } else if (a == "--track-max-age" && i + 1 < argc) {
        track_cfg.max_age = std::stoi(argv[++i]);
      } else if (a.rfind("--", 0) == 0) {
        std::cerr << "HATA: bilinmeyen veya degeri eksik secenek: " << a << "\n";
        return 1;
      } else if (positional == 0) {
        out_video_path = a;
        ++positional;
      } else if (positional == 1) {
        csv_path = a;
        ++positional;
      } else {
        std::cerr << "HATA: bilinmeyen veya fazla arguman: " << a << "\n";
        return 1;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "HATA: gecersiz sayisal arguman: " << error.what() << "\n";
    return 1;
  }
  if (!(conf_thr > 0.0f && conf_thr < 1.0f) ||
      !(nms_thr >= 0.0f && nms_thr <= 1.0f) ||
      (max_frames != -1 && max_frames <= 0)) {
    std::cerr << "HATA: --conf (0,1), --nms [0,1] ve --max-frames pozitif olmali\n";
    return 1;
  }
  if (track_cfg.n_init < 1 || track_cfg.max_age < 0) {
    std::cerr << "HATA: --track-n-init >= 1 ve --track-max-age >= 0 olmali\n";
    return 1;
  }
  // Guven esigi bilincli olarak dusuk (0.15); ByteTrack'in ikinci asamasinin
  // anlamli olmasi icin yuksek/dusuk siniri esigin uzerinde olmali.
  track_cfg.high_thr = std::max(conf_thr + 0.05f, 0.50f);
  tracking::Tracker tracker(track_cfg);

  // ---------------- DPU runner ----------------
  auto graph = xir::Graph::deserialize(model_path);
  auto children = graph->get_root_subgraph()->children_topological_sort();
  // **DPU subgraph sayisina** bakilir, toplama degil. Derlenmis graph'ta
  // 1 USER (girdi) + 1 DPU + cikti basina bir `fix2float` CPU blogu bulunur;
  // 9 ayri cikti tensoruyle bu 11 alt graph eder (2026-08-10'da kartta
  // olculdu). O CPU bloklari terminal dequant islemidir ve burada
  // kullanilmaz: asagida her cikti kendi `fix_point`'iyle host tarafinda
  // olceklenir. Tek DPU blogu varken agin ortasinda CPU'ya dusme yapisal
  // olarak imkansizdir, dolayisiyla dogru kapi budur.
  std::vector<xir::Subgraph*> dpu_subgraphs;
  for (auto* child : children) {
    if (child->has_attr("device") &&
        child->get_attr<std::string>("device") == "DPU") {
      dpu_subgraphs.push_back(child);
    }
  }
  if (dpu_subgraphs.size() != 1) {
    std::cerr << "HATA: xmodel tam olarak bir DPU subgraph icermeli; bulunan "
              << dpu_subgraphs.size() << " (toplam " << children.size()
              << " alt graph)\n";
    return 1;
  }
  xir::Subgraph* dpu_subgraph = dpu_subgraphs[0];
  auto attrs = xir::Attrs::create();
  auto runner = vart::RunnerExt::create_runner(dpu_subgraph, attrs.get());
  auto input_tbs = runner->get_inputs();
  auto output_tbs = runner->get_outputs();
  // 9 cikti = 3 stride seviyesi x (reg, obj, cls). Bas ciktilari bilerek
  // birlestirilmez; gerekcesi OutputLevel'in aciklamasinda.
  if (input_tbs.size() != 1 || output_tbs.size() != 9) {
    std::cerr << "HATA: beklenen tensor sayisi input=1/output=9 "
                 "(3 seviye x reg/obj/cls); bulunan "
              << input_tbs.size() << "/" << output_tbs.size() << "\n";
    return 1;
  }

  const xir::Tensor* in_tensor = input_tbs[0]->get_tensor();
  const auto in_shape = in_tensor->get_shape();
  if (in_shape.size() != 4 || in_shape[0] != 1 || in_shape[3] != 3) {
    std::cerr << "HATA: DPU girdisi NHWC [1,H,W,3] olmali\n";
    return 1;
  }
  const int in_h = in_shape.at(1);  // NHWC
  const int in_w = in_shape.at(2);
  if (!in_tensor->has_attr("fix_point")) {
    std::cerr << "HATA: input tensor fix_point niteligine sahip degil\n";
    return 1;
  }
  const float in_scale =
      std::exp2f(static_cast<float>(in_tensor->get_attr<int>("fix_point")));

  // sigmoid(obj) < conf hucrelerini ham int8 uzerinden ele (logit esigi)
  const float obj_logit_gate = -std::log(1.0f / conf_thr - 1.0f);

  // Rolleri kanal sayisindan tanimak icin ucunun de farkli olmasi sart.
  static_assert(kNumClassNames != 4 && kNumClassNames != 1,
                "Sinif sayisi 4 veya 1 olursa cikti rolleri kanal sayisindan "
                "ayirt edilemez; tensorleri ada gore eslemek gerekir.");

  const int num_classes = kNumClassNames;

  // VART cikti tensorlerinin SIRASINI GARANTI ETMEZ (Xilinx'in kendi
  // orneklerinde de bu yuzden acik bir output_mapping tutulur). Bu yuzden
  // siraya hic guvenmiyoruz: once uzamsal boyuta gore seviyelere gruplayip,
  // rolu kanal sayisindan taniyoruz (4=reg, 1=obj, num_classes=cls).
  std::vector<OutputLevel> levels;
  for (auto* tb : output_tbs) {
    const auto shape = tb->get_tensor()->get_shape();  // [1, H, W, C]
    if (shape.size() != 4 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 ||
        in_h % shape[1] != 0 || in_w % shape[2] != 0 ||
        in_h / shape[1] != in_w / shape[2] ||
        !tb->get_tensor()->has_attr("fix_point")) {
      std::cerr << "HATA: gecersiz YOLOX cikti tensor sekli/niteligi\n";
      return 1;
    }
    const int height = shape.at(1);
    const int width = shape.at(2);
    const int channels = shape.at(3);
    const float scale = std::exp2f(
        -1.0f * static_cast<float>(tb->get_tensor()->get_attr<int>("fix_point")));

    auto it = std::find_if(levels.begin(), levels.end(),
                           [&](const OutputLevel& lv) {
                             return lv.height == height && lv.width == width;
                           });
    if (it == levels.end()) {
      OutputLevel lv;
      lv.height = height;
      lv.width = width;
      lv.stride = in_h / height;
      lv.num_classes = num_classes;
      levels.push_back(lv);
      it = levels.end() - 1;
    }

    if (channels == 4) {
      it->reg = tb;
      it->reg_scale = scale;
    } else if (channels == 1) {
      it->obj = tb;
      it->obj_scale = scale;
    } else if (channels == num_classes) {
      it->cls = tb;
      it->cls_scale = scale;
    } else {
      std::cerr << "HATA: taninmayan cikti kanal sayisi " << channels
                << " (beklenen 4=reg, 1=obj, " << num_classes << "=cls)\n";
      return 1;
    }
  }

  if (levels.size() != 3) {
    std::cerr << "HATA: 3 stride seviyesi bekleniyordu; bulunan "
              << levels.size() << "\n";
    return 1;
  }
  std::sort(levels.begin(), levels.end(),
            [](const OutputLevel& a, const OutputLevel& b) {
              return a.stride < b.stride;
            });

  std::set<int> strides;
  for (auto& lv : levels) {
    if (!lv.complete()) {
      std::cerr << "HATA: stride " << lv.stride
                << " seviyesinde reg/obj/cls tensorlerinden biri eksik\n";
      return 1;
    }
    strides.insert(lv.stride);
    lv.reg_exp_lut.resize(256);
    lv.obj_sigmoid_lut.resize(256);
    lv.cls_sigmoid_lut.resize(256);
    for (int v = -128; v < 128; ++v) {
      const float f = static_cast<float>(v);
      lv.reg_exp_lut[v + 128] = std::exp(f * lv.reg_scale);
      lv.obj_sigmoid_lut[v + 128] = sigmoidf(f * lv.obj_scale);
      lv.cls_sigmoid_lut[v + 128] = sigmoidf(f * lv.cls_scale);
    }
    // obj kendi olcegine sahip oldugu icin gate de ona gore hesaplanir.
    lv.obj_gate = static_cast<int8_t>(std::max(
        -128.0f, std::min(127.0f, std::ceil(obj_logit_gate / lv.obj_scale))));
  }
  if (strides != std::set<int>({8, 16, 32})) {
    std::cerr << "HATA: cikti stride degerleri 8/16/32 olmali\n";
    return 1;
  }

  std::cout << "Model girisi : " << in_w << "x" << in_h
            << " (input_scale=" << in_scale << ")\n";
  std::cout << "Sinif sayisi : " << num_classes << "\n";
  for (const auto& lv : levels) {
    std::cout << "  cikti " << lv.width << "x" << lv.height
              << " stride=" << lv.stride
              << "  olcek reg=" << lv.reg_scale
              << " obj=" << lv.obj_scale
              << " cls=" << lv.cls_scale << "\n";
  }

  // ---------------- video girisi/ciktisi ----------------
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::cerr << "HATA: video acilamadi: " << video_path << "\n";
    return 1;
  }
  double fps_in = cap.get(cv::CAP_PROP_FPS);
  if (fps_in <= 1e-3) fps_in = 30.0;
  const int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

  cv::VideoWriter writer;
  if (draw_output) {
    writer.open(out_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                fps_in, cv::Size(frame_w, frame_h));
  }
  if (draw_output && !writer.isOpened()) {
    std::cerr << "UYARI: cikti videosu acilamadi (" << out_video_path
              << "); yalnizca CSV yazilacak\n";
  }

  std::ofstream csv(csv_path);
  if (!csv.is_open()) {
    std::cerr << "HATA: CSV dosyasi acilamadi: " << csv_path << "\n";
    return 1;
  }
  csv << "frame,track_id,class_id,class_name,score,cx,cy\n";
  csv << std::fixed << std::setprecision(2);

  // ---------------- kare dongusu ----------------
  cv::Mat frame;
  cv::Mat canvas(in_h, in_w, CV_8UC3);
  long frame_id = 0;
  long total_dets = 0;
  double sum_pre = 0.0, sum_dpu = 0.0, sum_post = 0.0;
  // Son-isleme dort ayri isi kapsiyor ve yogun sahnede hangisinin pahali
  // oldugu tahminle bilinemez: 2026-08-10'da seyrek sahnede 0,2 ms olan
  // son-isleme, ~370 araclik bir otoparkta 76 ms'ye cikti. Alt kirilim
  // olmadan optimizasyon korlemesine olur.
  double sum_decode = 0.0, sum_nms = 0.0, sum_track = 0.0, sum_draw = 0.0;
  const auto t_start = std::chrono::steady_clock::now();

  while (cap.read(frame)) {
    if (max_frames > 0 && frame_id >= max_frames) break;

    uint64_t in_data = 0u;
    size_t in_size = 0u;
    std::tie(in_data, in_size) =
        input_tbs[0]->data(std::vector<int>{0, 0, 0, 0});
    int8_t* in_ptr = reinterpret_cast<int8_t*>(in_data);
    const size_t expected_input =
        static_cast<size_t>(in_h) * in_w * 3 * sizeof(int8_t);
    if (in_ptr == nullptr || in_size < expected_input) {
      std::cerr << "HATA: DPU input buffer boyutu yetersiz\n";
      return 1;
    }

    // --- on isleme: YOLOX letterbox (sol-ust yerlesim, 114 dolgu, BGR 0-255)
    const auto t_pre = std::chrono::steady_clock::now();
    const float r = std::min(static_cast<float>(in_w) / frame.cols,
                             static_cast<float>(in_h) / frame.rows);
    const int nw = static_cast<int>(frame.cols * r);
    const int nh = static_cast<int>(frame.rows * r);
    canvas.setTo(cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(0, 0, nw, nh)));

    const uint8_t* src = canvas.data;
    const int n_px = in_h * in_w * 3;
    for (int i = 0; i < n_px; ++i) {
      const float v = static_cast<float>(src[i]) * in_scale;
      in_ptr[i] = static_cast<int8_t>(
          std::round(std::min(127.0f, std::max(-128.0f, v))));
    }
    for (auto* tb : input_tbs) {
      tb->sync_for_write(0, tb->get_tensor()->get_data_size() /
                                tb->get_tensor()->get_shape()[0]);
    }
    sum_pre += ms_since(t_pre);

    // --- DPU kosumu
    const auto t_dpu = std::chrono::steady_clock::now();
    auto job = runner->execute_async(input_tbs, output_tbs);
    if (runner->wait(static_cast<int>(job.first), -1) != 0) {
      std::cerr << "HATA: DPU isi basarisiz oldu\n";
      return 1;
    }
    for (auto* tb : output_tbs) {
      tb->sync_for_read(0, tb->get_tensor()->get_data_size() /
                               tb->get_tensor()->get_shape()[0]);
    }
    sum_dpu += ms_since(t_dpu);

    // --- son isleme: decode + NMS + merkez noktalari
    const auto t_post = std::chrono::steady_clock::now();
    if (frame_id == 0 && !dump_dir.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(dump_dir, ec);
      if (ec) {
        std::cerr << "HATA: golden dump klasoru olusturulamadi: "
                  << ec.message() << "\n";
        return 1;
      }
      std::ofstream meta(std::filesystem::path(dump_dir) / "meta.txt");
      if (!meta.is_open()) {
        std::cerr << "HATA: golden dump metadata dosyasi acilamadi\n";
        return 1;
      }
      meta << std::setprecision(10)
           << "input_h=" << in_h << "\n"
           << "input_w=" << in_w << "\n"
           << "input_scale=" << in_scale << "\n"
           << "input_file=input_int8.bin\n"
           << "canvas_file=canvas_bgr.bin\n"
           << "frame_h=" << frame.rows << "\n"
           << "frame_w=" << frame.cols << "\n"
           << "ratio=" << r << "\n"
           << "conf=" << conf_thr << "\n"
           << "nms=" << nms_thr << "\n"
           << "num_levels=" << levels.size() << "\n";
      {
        std::ofstream input_raw(
            std::filesystem::path(dump_dir) / "input_int8.bin",
            std::ios::binary);
        input_raw.write(reinterpret_cast<const char*>(in_ptr), expected_input);
        std::ofstream canvas_raw(
            std::filesystem::path(dump_dir) / "canvas_bgr.bin",
            std::ios::binary);
        canvas_raw.write(reinterpret_cast<const char*>(canvas.data),
                         expected_input);
        if (!input_raw || !canvas_raw) {
          std::cerr << "HATA: golden dump input dosyalari yazilamadi\n";
          return 1;
        }
      }
      for (size_t level_idx = 0; level_idx < levels.size(); ++level_idx) {
        const auto& lv = levels[level_idx];
        const size_t cells = static_cast<size_t>(lv.height) * lv.width;
        const struct {
          const char* role;
          vart::TensorBuffer* tb;
          int channels;
          float scale;
        } parts[] = {
            {"reg", lv.reg, 4, lv.reg_scale},
            {"obj", lv.obj, 1, lv.obj_scale},
            {"cls", lv.cls, lv.num_classes, lv.cls_scale},
        };
        meta << "level" << level_idx << "_h=" << lv.height << "\n"
             << "level" << level_idx << "_w=" << lv.width << "\n"
             << "level" << level_idx << "_stride=" << lv.stride << "\n"
             << "level" << level_idx << "_nc=" << lv.num_classes << "\n";
        for (const auto& part : parts) {
          const size_t expected = cells * part.channels * sizeof(int8_t);
          const int8_t* raw_ptr =
              tensor_data(part.tb, expected, "golden dump output");
          if (raw_ptr == nullptr) return 1;
          const std::string file_name = "output_stride" +
                                        std::to_string(lv.stride) + "_" +
                                        part.role + ".bin";
          std::ofstream raw(std::filesystem::path(dump_dir) / file_name,
                            std::ios::binary);
          raw.write(reinterpret_cast<const char*>(raw_ptr), expected);
          if (!raw) {
            std::cerr << "HATA: golden dump tensoru yazilamadi\n";
            return 1;
          }
          meta << "level" << level_idx << "_" << part.role
               << "_file=" << file_name << "\n"
               << "level" << level_idx << "_" << part.role
               << "_scale=" << part.scale << "\n";
        }
      }
    }
    std::vector<Detection> cands;
    for (const auto& lv : levels) {
      const size_t cells = static_cast<size_t>(lv.height) * lv.width;
      const int8_t* reg = tensor_data(lv.reg, cells * 4, "reg");
      const int8_t* obj = tensor_data(lv.obj, cells * 1, "obj");
      const int8_t* cls = tensor_data(lv.cls, cells * lv.num_classes, "cls");
      if (reg == nullptr || obj == nullptr || cls == nullptr) return 1;

      for (int gy = 0; gy < lv.height; ++gy) {
        for (int gx = 0; gx < lv.width; ++gx) {
          const int cell = gy * lv.width + gx;
          const int8_t obj_raw = obj[cell];
          if (obj_raw < lv.obj_gate) continue;  // hizli eleme
          const float obj_s = lv.obj_sigmoid_lut[obj_raw + 128];

          // en yuksek sinif: sigmoid monoton oldugu icin ham deger yeter
          const int cls_base = cell * lv.num_classes;
          int best_c = 0;
          int8_t best_raw = cls[cls_base];
          for (int c = 1; c < lv.num_classes; ++c) {
            if (cls[cls_base + c] > best_raw) {
              best_raw = cls[cls_base + c];
              best_c = c;
            }
          }
          const float score = obj_s * lv.cls_sigmoid_lut[best_raw + 128];
          if (score < conf_thr) continue;

          // YOLOX decode (reg ciktilari ham, sigmoid'siz)
          const int reg_base = cell * 4;
          const float cx =
              (static_cast<float>(reg[reg_base + 0]) * lv.reg_scale + gx) *
              lv.stride;
          const float cy =
              (static_cast<float>(reg[reg_base + 1]) * lv.reg_scale + gy) *
              lv.stride;
          const float bw = lv.reg_exp_lut[reg[reg_base + 2] + 128] * lv.stride;
          const float bh = lv.reg_exp_lut[reg[reg_base + 3] + 128] * lv.stride;

          Detection det;
          det.x1 = cx - bw * 0.5f;
          det.y1 = cy - bh * 0.5f;
          det.x2 = cx + bw * 0.5f;
          det.y2 = cy + bh * 0.5f;
          det.score = score;
          det.cls = best_c;
          cands.push_back(det);
        }
      }
    }
    sum_decode += ms_since(t_post);
    const auto t_nms = std::chrono::steady_clock::now();
    std::vector<Detection> dets = nms_per_class(cands, nms_thr);
    sum_nms += ms_since(t_nms);
    std::ofstream golden_csv;
    if (frame_id == 0 && !dump_dir.empty()) {
      golden_csv.open(std::filesystem::path(dump_dir) / "cpp_detections.csv");
      if (!golden_csv.is_open()) {
        std::cerr << "HATA: golden dump tespit dosyasi acilamadi\n";
        return 1;
      }
      golden_csv << "class_id,score,x1,y1,x2,y2,cx,cy\n"
                 << std::setprecision(10);
    }
    long valid_frame_dets = 0;

    // Tespitleri orijinal video koordinatlarina tasi. Golden dump ham
    // tespitleri yazar (takipten once), boylece Python-C++ esdegerlik testi
    // takip katmanindan etkilenmez.
    std::vector<tracking::Observation> observations;
    observations.reserve(dets.size());
    for (const auto& det : dets) {
      // letterbox olceginden orijinal video koordinatlarina don
      float x1 = std::max(0.0f, std::min(det.x1 / r, frame.cols - 1.0f));
      float y1 = std::max(0.0f, std::min(det.y1 / r, frame.rows - 1.0f));
      float x2 = std::max(0.0f, std::min(det.x2 / r, frame.cols - 1.0f));
      float y2 = std::max(0.0f, std::min(det.y2 / r, frame.rows - 1.0f));
      if (x2 <= x1 || y2 <= y1) continue;

      if (golden_csv.is_open()) {
        // ---- MERKEZ NOKTASI ----
        golden_csv << det.cls << "," << det.score << "," << x1 << "," << y1
                   << "," << x2 << "," << y2 << "," << (x1 + x2) * 0.5f << ","
                   << (y1 + y2) * 0.5f << "\n";
      }
      ++valid_frame_dets;
      observations.push_back({{x1, y1, x2, y2}, det.score, det.cls});
    }

    // Cizilecek/loglanacak nesneler: takip aciksa onaylanmis izler, kapaliysa
    // ham tespitler (iz kimligi -1).
    struct Rendered {
      tracking::Box box;
      float score;
      int cls;
      int track_id;
    };
    std::vector<Rendered> rendered;
    const auto t_track = std::chrono::steady_clock::now();
    if (use_tracker) {
      for (const tracking::Track& t : tracker.update(observations)) {
        if (t.visible()) rendered.push_back({t.box, t.score, t.cls, t.id});
      }
    } else {
      for (const auto& o : observations) {
        rendered.push_back({o.box, o.score, o.cls, -1});
      }
    }
    sum_track += ms_since(t_track);

    const auto t_draw = std::chrono::steady_clock::now();
    for (const Rendered& item : rendered) {
      // ---- MERKEZ NOKTASI ----
      const float cx = item.box.cx();
      const float cy = item.box.cy();
      const char* cname = (item.cls >= 0 && item.cls < kNumClassNames)
                              ? kClassNames[item.cls]
                              : "?";
      csv << frame_id << "," << item.track_id << "," << item.cls << ","
          << cname << "," << item.score << "," << cx << "," << cy << "\n";

      if (!draw_output) continue;
      cv::rectangle(
          frame,
          cv::Point(static_cast<int>(item.box.x1), static_cast<int>(item.box.y1)),
          cv::Point(static_cast<int>(item.box.x2), static_cast<int>(item.box.y2)),
          cv::Scalar(0, 255, 0), 2);
      cv::circle(frame, cv::Point(static_cast<int>(cx), static_cast<int>(cy)), 3,
                 cv::Scalar(0, 0, 255), -1);
      char label[96];
      if (item.track_id >= 0) {
        std::snprintf(label, sizeof(label), "#%d %s %.2f (%d,%d)",
                      item.track_id, cname, item.score, static_cast<int>(cx),
                      static_cast<int>(cy));
      } else {
        std::snprintf(label, sizeof(label), "%s %.2f (%d,%d)", cname,
                      item.score, static_cast<int>(cx), static_cast<int>(cy));
      }
      cv::putText(frame, label,
                  cv::Point(static_cast<int>(item.box.x1),
                            std::max(12, static_cast<int>(item.box.y1) - 4)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
    }
    sum_draw += ms_since(t_draw);
    total_dets += valid_frame_dets;
    sum_post += ms_since(t_post);

    if (writer.isOpened()) writer.write(frame);
    ++frame_id;
    if (frame_id % 50 == 0) {
      std::cout << "kare " << frame_id << " islendi (" << valid_frame_dets
                << " tespit, " << rendered.size()
                << (use_tracker ? " onayli iz)" : " cizim)") << "\n";
    }
  }

  const double total_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
          .count();
  if (frame_id == 0) {
    std::cerr << "HATA: videodan hic kare okunamadi\n";
    return 1;
  }
  std::cout << "\n==== Ozet ====\n";
  std::cout << "kare sayisi      : " << frame_id << "\n";
  std::cout << "toplam tespit    : " << total_dets << "\n";
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "ort. on-isleme   : " << sum_pre / frame_id << " ms\n";
  std::cout << "ort. DPU         : " << sum_dpu / frame_id << " ms\n";
  std::cout << "ort. son-isleme  : " << sum_post / frame_id << " ms\n";
  std::cout << std::setprecision(2);
  std::cout << "     decode      : " << sum_decode / frame_id << " ms\n";
  std::cout << "     NMS         : " << sum_nms / frame_id << " ms\n";
  std::cout << "     takip       : " << sum_track / frame_id << " ms\n";
  std::cout << "     cizim       : " << sum_draw / frame_id << " ms"
            << (draw_output ? "\n" : "  (--no-video ile kapali)\n");
  std::cout << std::setprecision(1);
  std::cout << "uctan uca FPS    : " << frame_id / total_s << "\n";
  std::cout << "cikti video      : " << (writer.isOpened() ? out_video_path : "(yok)")
            << "\n";
  std::cout << "merkez log (CSV) : " << csv_path << "\n";
  return 0;
}
