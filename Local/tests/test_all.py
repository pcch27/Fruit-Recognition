# -*- coding: utf-8 -*-
"""不需要 torch、GPU、網路就能跑的測試。執行方式：python tests/test_all.py"""
import ast
import csv
import json
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "B"))
import a_common as ac
import yolo_data as yd


def save_image(path, seed, size=(60, 40)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def run_script(args, cwd):
    result = subprocess.run([sys.executable] + args, capture_output=True, text=True, cwd=cwd)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_parse_filenames():
    assert ac.tokens_from_filename("apple_banana_lemon_1.jpg") == ["apple", "banana", "lemon"]
    assert ac.tokens_from_filename("cherry(wax)_3.jpg") == ["cherry"]
    assert ac.tokens_from_filename("Banana(lady_finger)_2.jpg") == ["banana"]
    assert ac.tokens_from_filename("ananas_1.jpg") == ["pineapple"]
    assert ac.tokens_from_filename("apple_apricot_peach_peach(flat)_pear_plum_pomegranate_2.jpg") == \
        ["apple", "apricot", "peach", "pear", "plum", "pomegranate"]
    assert ac.tokens_from_filename("cactus fruit.jpg") == ["cactus"]
    assert ac.tokens_from_filename("cherry_sour_1.jpg") == ["cherry"]          # sour 只是形容，不是水果
    assert ac.tokens_from_filename("peanut shell_1.jpg") == ["peanut"]         # 堅果 -> A1 會略過
    assert ac.tokens_from_filename("potato_sweet_1.jpg") == ["potato"]
    assert ac.photo_number("apple_105.JPG") == 105 and ac.photo_number("blackberry.jpg") == 0
    assert ac.composition_key("apple_banana_lemon_1.jpg") == "apple_banana_lemon"
    assert ac.family_from_class_name("Grape White 3") == "grape"
    # original-size 有些資料夾用底線分開（這是之前的錯誤：會變成很多個「蘋果」）
    assert ac.family_from_class_name("apple_braeburn_1") == "apple"
    assert ac.family_from_class_name("cabbage_white_1") == "cabbage" and not ac.is_fruit("cabbage")
    assert ac.family_from_class_name("zucchini_1") == "zucchini" and ac.family_from_class_name("Cherry Wax Red 1") == "cherry"
    assert ac.is_fruit("apple") and not ac.is_fruit("pepper") and not ac.is_fruit("tomato")


def test_split_groups():
    groups = {"apple": ["apple|%d" % i for i in range(20)], "rare": ["rare|0", "rare|1"]}
    result = ac.split_groups(groups, 1)
    assert result["rare|0"] == "train" and result["rare|1"] == "train"
    kinds = set(result[g] for g in groups["apple"])
    assert kinds == {"train", "val", "test"}
    again = ac.split_groups(groups, 1)
    assert result == again
    assert sum(1 for g in groups["apple"] if result[g] == "test") == 3


def test_metrics():
    scores = [0.1, 0.9, 0.3, 0.8]
    assert ac.top_indices(scores, 2) == [1, 3]
    assert ac.top1_hit(scores, [1]) and not ac.top1_hit(scores, [3])
    assert ac.topk_hit(scores, [3], 2) and not ac.topk_hit(scores, [0], 2)
    assert ac.found_ratio(scores, [1, 3]) == 1.0
    assert ac.found_ratio(scores, [1, 2]) == 0.5


def make_fake_fruits360(root):
    """假的 Fruits-360：100x100 與 original-size，各有水果與蔬菜類別。"""
    classes100 = ["Apple Red 1", "Apple Golden 1", "Cherry 1", "Cherry Wax Red 1", "Redcurrant 1", "Pepper Green 1", "Tomato 1", "Almonds 1"]
    for split, n in (("Training", 6), ("Test", 3)):
        for c in classes100:
            for k in range(n):
                save_image(os.path.join(root, "f100", split, c, str(k) + "_100.jpg"), k + len(c), (20, 20))
    classes_ori = ["Apple Red", "Cherry 1", "Pear Forelle", "Corn", "apple_braeburn_1", "cabbage_white_1"]
    for split, n in (("Training", 4), ("Validation", 2), ("Test", 2)):
        for c in classes_ori:
            for k in range(n):
                save_image(os.path.join(root, "ori", split, c, str(k) + ".jpg"), k + len(c), (30, 30))


def make_fake_multi(folder):
    names = ["apple_%d.jpg" % i for i in range(1, 41)]
    names += ["cherry_%d.jpg" % i for i in range(1, 25)]
    names += ["currant_1.jpg", "carrot_1.jpg", "carrot_2.jpg", "pepper_3.jpg", "pear_1.jpg"]
    names += ["apple_banana_lemon_1.jpg", "apple_banana_lemon_2.jpg", "apple_pear_pepper.jpg", "apple_pear_1.jpg",
              "apple_pear_2.jpg", "cherry(wax)_apple_1.jpg", "apple_cherry.jpg", "pear_cherry_1.jpg", "pear_cherry_2.jpg"]
    for i, name in enumerate(names):
        save_image(os.path.join(folder, name), i, (200, 120))
    open(os.path.join(folder, "list.txt"), "w").write("not an image")
    return names


def test_A1_A2_pipeline():
    with tempfile.TemporaryDirectory() as td:
        fruits = os.path.join(td, "fruits")
        make_fake_fruits360(fruits)
        multi = os.path.join(td, "multi")
        make_fake_multi(multi)
        out_real = os.path.join(td, "real")
        text = run_script([os.path.join(ROOT, "A", "A1_prepare_real_photos.py"), "--src", multi, "--out", out_real, "--max_side", "64"], ROOT)
        assert "略過" in text
        rows = list(csv.DictReader(open(os.path.join(out_real, "real_manifest.csv"), encoding="utf-8")))
        names = [r["file"] for r in rows]
        assert "carrot_1.jpg" not in names and "pepper_3.jpg" not in names     # 只有蔬菜 -> 不要
        by_name = {r["file"]: r for r in rows}
        assert by_name["apple_pear_pepper.jpg"]["families"] == "apple|pear"  # 蔬菜字被拿掉
        assert by_name["cherry_1.jpg"]["families"] == "cherry"
        assert by_name["cherry(wax)_apple_1.jpg"]["families"] == "cherry|apple"  # 括號內的字（wax）被拿掉
        # 同組不跨 split：apple 的連號每 8 張一組
        for r in rows:
            if r["families"] == "apple":
                same = [x for x in rows if x["group"] == r["group"]]
                assert len(set(x["split"] for x in same)) == 1
        # 縮圖有存、長邊不超過 64
        for r in rows[:5]:
            im = Image.open(os.path.join(out_real, "images", r["file"]))
            assert max(im.size) <= 64
        # A2
        out_lists = os.path.join(td, "lists")
        text = run_script([os.path.join(ROOT, "A", "A2_build_lists.py"), "--fruits100", os.path.join(fruits, "f100"),
                           "--original", os.path.join(fruits, "ori"), "--real_dir", out_real, "--out", out_lists], ROOT)
        families = json.load(open(os.path.join(out_lists, "families.json"), encoding="utf-8"))
        assert families == ["apple", "cherry", "pear", "redcurrant"], families
        white = json.load(open(os.path.join(out_lists, "white_lists.json"), encoding="utf-8"))
        assert set(white.keys()) == {"100_training", "100_test", "original_training", "original_validation", "original_test"}
        assert len(white["100_training"]) == 6 * 5      # apple red, apple golden, cherry 1, cherry wax red, redcurrant（蔬菜、堅果不算）
        assert all(0 <= item[1] < 4 for item in white["original_training"])
        assert len(white["original_training"]) == 4 * 4   # apple red, cherry 1, pear forelle, apple_braeburn_1（玉米、高麗菜不算）
        apple_id = families.index("apple")
        assert sum(1 for item in white["original_training"] if item[1] == apple_id) == 8   # 底線寫法的蘋果也歸到 apple
        real = json.load(open(os.path.join(out_lists, "real_lists.json"), encoding="utf-8"))
        total = len(real["train"]) + len(real["val"]) + len(real["test"])
        assert total == len(rows) - 0
        # currant 檔名會對應到 redcurrant（大類清單裡有）
        redcurrant_id = families.index("redcurrant")
        assert any(item[1] == [redcurrant_id] for part in real.values() for item in part)
        # banana、lemon 不在大類清單 -> 被忽略，但 apple 還在
        multi_items = [item for part in real.values() for item in part if "apple_banana_lemon" in item[0]]
        assert len(multi_items) == 2 and all(item[1] == [families.index("apple")] for item in multi_items)
        assert "沒有真實照片" not in text      # 假資料裡每個大類都有真實照片
        assert "apple" in text and "multi" in text


def make_fake_yolo(root):
    rng = np.random.default_rng(0)
    for split, n in (("train", 10), ("val", 3), ("test", 3)):
        for k in range(n):
            image_dir = os.path.join(root, "images", split)
            label_dir = os.path.join(root, "labels", split)
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            save_image(os.path.join(image_dir, split + "_" + str(k) + ".jpg"), k + len(split), (160, 120))
            lines = []
            for j in range(int(rng.integers(1, 4))):
                lines.append("%d %.4f %.4f 0.2 0.25" % (int(rng.integers(0, 3)), rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)))
            if split == "train" and k == 0:
                lines.append("9 0.5 0.5 0.1 0.1")
            with open(os.path.join(label_dir, split + "_" + str(k) + ".txt"), "w") as f:
                f.write("\n".join(lines))
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: Apple\n  1: Banana\n  2: Mango\n")


def test_B1_and_helpers():
    with tempfile.TemporaryDirectory() as td:
        make_fake_yolo(td)
        yaml_path, data = yd.find_yaml(td)
        assert yd.load_names(data) == ["Apple", "Banana", "Mango"]
        for s in ("train", "val", "test"):
            image_dir, label_dir = yd.find_split_dirs(td, s)
            assert image_dir is not None and os.path.isdir(label_dir)
        boxes, problems = yd.read_labels(os.path.join(td, "labels", "train", "train_0.txt"), 3)
        assert "類別編號超出範圍" in problems
        assert yd.counts_by_class([0.9, 0.3, 0.6], [0, 0, 2], 0.5, 3) == [1, 0, 1]
        text = run_script([os.path.join(ROOT, "B", "B1_check_dataset.py"), "--root", td, "--sanity", "6", "--out", os.path.join(td, "sanity")], td)
        assert "類別數： 3" in text and "[train]" in text and "類別編號超出範圍" in text
        assert len(os.listdir(os.path.join(td, "sanity"))) >= 3


def test_eval_utils():
    sys.path.insert(0, os.path.join(ROOT, "B"))
    import b_eval_utils as be
    assert be.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert be.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(be.iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1.0 / 3.0) < 1e-9
    assert be.yolo_to_xyxy(0.5, 0.5, 0.4, 0.4, 200, 100) == (60.0, 30.0, 140.0, 70.0)
    assert be.family_of_name("Apple Red") == "apple"
    gt = [{"family": "apple", "box": (0, 0, 10, 10)}, {"family": "pear", "box": (50, 50, 80, 80)}]
    pred = [{"name": "Apple Red", "conf": 0.9, "box": (0, 0, 10, 10)},
            {"name": "Banana", "conf": 0.8, "box": (50, 50, 80, 80)},       # 位置對、種類錯
            {"name": "Apple Red", "conf": 0.7, "box": (100, 100, 120, 120)}]  # 多出來
    matched, class_ok, false_pos, missed = be.match_boxes(gt, pred, 0.5)
    assert (matched, class_ok, false_pos, missed) == (2, 1, 1, 0)
    # 一個標註框只能被配一次
    matched, class_ok, false_pos, missed = be.match_boxes([gt[0]], [pred[0], dict(pred[0])], 0.5)
    assert (matched, false_pos, missed) == (1, 1, 0)


def test_B4_picker():
    with tempfile.TemporaryDirectory() as td:
        multi = os.path.join(td, "multi")
        make_fake_multi(multi)
        real = os.path.join(td, "real")
        run_script([os.path.join(ROOT, "A", "A1_prepare_real_photos.py"), "--src", multi, "--out", real, "--max_side", "64"], ROOT)
        out = os.path.join(td, "b_test")
        text = run_script([os.path.join(ROOT, "B", "B4_pick_photos.py"), "--real_dir", real, "--out", out,
                           "--n_covered", "3", "--n_not_covered", "3", "--n_mixed", "3"], ROOT)
        manifest = {}
        for row in csv.DictReader(open(os.path.join(real, "real_manifest.csv"), encoding="utf-8")):
            manifest[row["file"]] = row
        picked = list(csv.DictReader(open(os.path.join(out, "picked_list.csv"), encoding="utf-8")))
        assert len(picked) >= 3
        for row in picked:
            assert manifest[row["file"]]["split"] == "test", "只能從 test 挑"
            assert os.path.exists(os.path.join(out, "photos", row["file"]))
            names = row["families"].split("|")
            if row["group"] == "mixed":
                assert len(names) >= 2
            elif row["group"] == "covered":
                assert len(names) == 1 and names[0] in ("apple", "pear", "banana", "kiwi", "lemon", "mango", "orange")
            else:
                assert row["group"] == "not_covered" and len(names) == 1 and names[0] == "cherry"
        classes = [line.strip() for line in open(os.path.join(out, "classes.txt"), encoding="utf-8").read().splitlines()]
        assert classes == sorted(set(classes)) and "apple" in classes
        assert "不要拿去訓練 B" in text


STUB_TORCH = """
class _Cuda:
    @staticmethod
    def is_available():
        return False
cuda = _Cuda()
"""

STUB_ULTRALYTICS = """
import os, types

class _T(object):
    def __init__(self, values):
        self.values = values
    def tolist(self):
        return self.values

class _Box(object):
    def __init__(self, items):
        self.xyxy = _T([i[0] for i in items])
        self.conf = _T([i[1] for i in items])
        self.cls = _T([i[2] for i in items])
    def __len__(self):
        return len(self.conf.values)

class _Result(object):
    def __init__(self, items):
        self.boxes = _Box(items)
        self.names = {0: "Apple Red", 1: "Banana", 2: "Tomato"}

class _BoxMetrics(object):
    mp = 0.8
    mr = 0.7
    map50 = 0.75
    map = 0.5
    ap_class_index = [0, 1]
    ap50 = [0.9, 0.6]
    ap = [0.6, 0.4]

class _Metrics(object):
    box = _BoxMetrics()

class YOLO(object):
    def __init__(self, path):
        self.path = path
        self.names = {0: "Apple Red", 1: "Banana", 2: "Tomato"}
    def train(self, **kwargs):
        save_dir = os.path.join(os.getcwd(), "runs_stub", kwargs.get("name", "x"))
        os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)
        with open(os.path.join(save_dir, "weights", "best.pt"), "w") as f:
            f.write("stub")
        self.trainer = types.SimpleNamespace(save_dir=save_dir)
    def val(self, **kwargs):
        return _Metrics()
    def predict(self, source=None, conf=0.25, **kwargs):
        items = [((60, 30, 140, 70), 0.9, 0), ((5, 5, 20, 20), 0.6, 1), ((150, 60, 190, 95), 0.8, 2)]
        if isinstance(source, list):
            return (_Result(items) for _ in source)
        return [_Result(items)]
"""


def run_with_stubs(args, cwd, stub_dir):
    env = dict(os.environ)
    env["PYTHONPATH"] = stub_dir
    result = subprocess.run([sys.executable] + args, capture_output=True, text=True, cwd=cwd, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_B2_B3_B5_dry_run_with_stubs():
    """用「假的 torch 與 ultralytics」把 B2、B3、B5 整個流程跑一遍，檢查程式本身的邏輯（不是真的訓練）。"""
    with tempfile.TemporaryDirectory() as td:
        stub_dir = os.path.join(td, "stubs")
        os.makedirs(stub_dir)
        open(os.path.join(stub_dir, "torch.py"), "w").write(STUB_TORCH)
        open(os.path.join(stub_dir, "ultralytics.py"), "w").write(STUB_ULTRALYTICS)
        data = os.path.join(td, "yolo")
        make_fake_yolo(data)
        work = os.path.join(td, "work")
        os.makedirs(work)
        text = run_with_stubs([os.path.join(ROOT, "B", "B2_train_yolo.py"), "--data", os.path.join(data, "data.yaml"), "--epochs", "1"], work, stub_dir)
        assert os.path.exists(os.path.join(work, "models", "b_yolo.pt"))
        assert "Apple Red" in text and "AP50" in text
        summary = json.load(open(os.path.join(work, "results", "b_yolo_summary.json"), encoding="utf-8"))
        assert abs(summary["val"]["mAP50"] - 0.75) < 1e-9 and "Banana" in summary["test"]["per_class"]
        text = run_with_stubs([os.path.join(ROOT, "B", "B3_count_eval.py"), "--weights", "w.pt", "--root", data], work, stub_dir)
        assert "信心門檻" in text and os.path.exists(os.path.join(work, "results", "b_counts.json"))
        # B5：一張 covered 照片，標註 1 顆蘋果（和假偵測第一個框完全重疊）
        photos = os.path.join(td, "photos")
        save_image(os.path.join(photos, "p1.jpg"), 1, (200, 100))
        with open(os.path.join(photos, "p1.txt"), "w") as f:
            f.write("0 0.5 0.5 0.4 0.4\n")
        with open(os.path.join(td, "classes.txt"), "w") as f:
            f.write("apple\n")
        with open(os.path.join(td, "picked.csv"), "w") as f:
            f.write("file,group,families\np1.jpg,covered,apple\n")
        text = run_with_stubs([os.path.join(ROOT, "B", "B5_eval_real.py"), "--weights", "w.pt", "--photos", photos,
                               "--classes", os.path.join(td, "classes.txt"), "--groups", os.path.join(td, "picked.csv")], work, stub_dir)
        report = json.load(open(os.path.join(work, "results", "b_real_eval.json"), encoding="utf-8"))
        r = report["covered"]
        # 偵測：蘋果（對）、香蕉（誤框）、番茄（蔬菜，被忽略） -> 偵測框 2、配對 1
        assert r["images"] == 1 and r["gt_boxes"] == 1 and r["pred_boxes"] == 2
        assert r["recall"] == 1.0 and r["precision"] == 0.5 and r["class_correct_rate"] == 1.0 and r["count_mae"] == 1.0


def test_A4_real_single_report():
    """用假的 torch 與假的 predict_scores，檢查 A4 新增的「每個水果正確率 + 混淆」報告程式。"""
    import importlib.util
    fake_torch = types.ModuleType("torch")
    fake_du = types.ModuleType("a_data_utils")
    # 3 個類別、5 張單一種類照片：真實 [0,0,1,1,2]，模型第一名 [0,1,1,1,0]
    table = [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.0, 0.9, 0.1], [0.8, 0.1, 0.3]]
    fake_du.predict_scores = lambda model, items, n, device, batch, workers: np.array(table)
    saved = {}
    for name, module in (("torch", fake_torch), ("a_data_utils", fake_du)):
        saved[name] = sys.modules.get(name)
        sys.modules[name] = module
    try:
        spec = importlib.util.spec_from_file_location("a4_module", os.path.join(ROOT, "A", "A4_evaluate.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        items = [["a.jpg", [0]], ["b.jpg", [0]], ["c.jpg", [1]], ["d.jpg", [1]], ["e.jpg", [2]]]
        with tempfile.TemporaryDirectory() as td:
            old = os.getcwd()
            os.chdir(td)
            try:
                module.real_single_report(None, items, ["apple", "pear", "lemon"], "cpu", 8, 0, "demo")
                rows = list(csv.reader(open(os.path.join(td, "results", "demo_per_family.csv"), encoding="utf-8")))
                assert rows[0] == ["水果", "張數", "第一名答對", "正確率"]
                by_name = {r[0]: r for r in rows[1:]}
                assert by_name["apple"][1:4] == ["2", "1", "0.5"] and by_name["pear"][1:4] == ["2", "2", "1.0"]
                assert by_name["lemon"][1:4] == ["1", "0", "0.0"]
                assert os.path.exists(os.path.join(td, "results", "demo_confusion_real.png"))
            finally:
                os.chdir(old)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_data_utils_pure_logic():
    """a_data_utils 需要 torch；這裡用「假的 torch」只測不依賴 torch 的函式。"""
    import random
    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.BatchNorm2d = type("BatchNorm2d", (), {})
    fake_utils = types.ModuleType("torch.utils")
    fake_data = types.ModuleType("torch.utils.data")
    fake_data.Dataset = object
    fake_data.DataLoader = object
    fake_torch.nn = fake_nn
    fake_torch.utils = fake_utils
    fake_utils.data = fake_data
    saved = {}
    for name, module in (("torch", fake_torch), ("torch.nn", fake_nn), ("torch.utils", fake_utils), ("torch.utils.data", fake_data)):
        saved[name] = sys.modules.get(name)
        sys.modules[name] = module
    try:
        sys.modules.pop("a_data_utils", None)
        import a_data_utils as du
        by_class = du.group_by_class([["a.jpg", 0], ["b.jpg", 0], ["c.jpg", 1]])
        assert by_class == {0: ["a.jpg", "b.jpg"], 1: ["c.jpg"]}
        picked = du.pick_white(by_class, 1, random.Random(1))
        assert len(picked) == 2 and all(item[2] == "white" for item in picked)
        assert sorted(item[1][0] for item in picked) == [0, 1]
        picked_n = du.pick_white_n(by_class, 2, random.Random(1))
        assert len(picked_n) == 2
        assert du.val_score({"single_n": 4, "single_top1": 0.5, "multi_n": 2, "multi_found_ratio": 0.7}) == 0.6
        assert du.val_score({"single_n": 0, "multi_n": 0}) == 0.0
    finally:
        sys.modules.pop("a_data_utils", None)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_notebooks_are_valid():
    for name in ("Colab_A.ipynb", "Colab_B.ipynb"):
        path = os.path.join(os.path.dirname(ROOT), "Colab", name)
        data = json.load(open(path, encoding="utf-8"))
        assert data["nbformat"] == 4 and len(data["cells"]) >= 8
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                text = "".join(cell["source"])
                for line in text.splitlines():
                    if not line.strip().startswith("!") and not line.strip().startswith("%"):
                        pass
                # 不能有 f-string（Colab_B.ipynb 是從 Colab 匯出的，內含 f-string，不檢查）
                if name == "Colab_A.ipynb":
                    assert 'f"' not in text and "f'" not in text, (name, text[:80])


def test_no_fstrings():
    """全部 .py 都不能用 f-string（照使用者要求，寫法要淺顯）。"""
    bad = []
    for folder, dirs, files in os.walk(ROOT):
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                tree = ast.parse(open(path, encoding="utf-8").read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.JoinedStr):
                        bad.append(path + ":" + str(node.lineno))
    assert bad == [], bad


if __name__ == "__main__":
    names = sorted(k for k in globals() if k.startswith("test_"))
    for name in names:
        globals()[name]()
        print("通過", name)
    print(len(names), "項測試通過")
