# deim-steel

DEIMv2 (upstream `Intellindust-AI-Lab/DEIMv2`, коммит `1d2ca42`) для детекции дефектов стали. Здесь один
конфиг на весь эксперимент, загрузка всех весов одной командой и исправления для Windows, RTX 50xx и
прямоугольного входа. Описание самого DEIMv2 — в [README_DEIMv2.md](README_DEIMv2.md). Лицензия DEIMv2
([LICENSE.md](LICENSE.md)) допускает только некоммерческое использование. Веса DINOv3, включая backbone
внутри чекпоинта DEIMv2, распространяются по [LICENSE_DINOv3.md](LICENSE_DINOv3.md).

## Установка

```bash
uv sync          # .venv по pyproject.toml и uv.lock: torch 2.7.1 со сборкой CUDA 12.8 (нужна для RTX 50xx)
```

Скрипты `*.sh` сами находят `.venv` репозитория. Другой интерпретатор задаётся через
`PYTHON=/путь/к/python bash train.sh`. Без uv:
`pip install -r requirements_steel.txt` (тот же набор пакетов). На старых картах подойдёт исходный
`requirements.txt` с torch 2.5.1.

## 1. Веса

Веса для DEIMv2-X уже лежат в репозитории через Git LFS: `weights/deimv2_dinov3_x_coco.pth` (196 МБ) и
`ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` (110 МБ). Чтобы получить сами файлы, а не указатели:

```bash
git lfs install      # один раз; git-lfs: apt install git-lfs / conda install -c conda-forge git-lfs
git lfs pull         # или сразу: git lfs clone https://github.com/PTyneu/deim-steel
```

Скрипт загрузки тоже распознаёт указатели LFS и сначала сам вызывает `git lfs pull`. Для остальных размеров
и при недоступном LFS веса скачиваются из сети:

```bash
bash scripts/download_weights.sh             # веса для модели из experiment.yml
bash scripts/download_weights.sh x           # конкретный размер: s | m | l | x | all
bash scripts/download_weights.sh x --no-backbone
bash scripts/download_weights.sh x --source hf   # только Hugging Face (если Google Drive недоступен)
```

| Что | Куда | Откуда |
|---|---|---|
| Чекпоинт DEIMv2, обученный на COCO (стартовая точка дообучения) | `weights/deimv2_dinov3_<m>_coco.pth` | Google Drive авторов DEIMv2; при ошибке автоматически Hugging Face (`Intellindust/DEIMv2_DINOv3_<M>_COCO`, те же веса бит в бит) |
| Backbone для S и M: ViT-Tiny, дистиллированный из DINOv3 | `ckpts/vitt_distill.pt`, `ckpts/vittplus_distill.pt` | Google Drive авторов DEIMv2 |
| Backbone для L и X: DINOv3 ViT-S16 и ViT-S16+ | `ckpts/dinov3_vits16*_pretrain_lvd1689m-*.pth` | веса timm на Hugging Face без заявки у Meta; конвертируются в официальный формат с проверкой, выход совпадает с timm бит в бит |

Backbone нужен только для обучения с нуля (`weights: none`) или по upstream-конфигам. При дообучении с
COCO-чекпоинта все веса берутся из него.

Если сеть ограничена (сервер без доступа к Google или Hugging Face):
- **зеркало Hugging Face:** `HF_ENDPOINT=https://hf-mirror.com bash scripts/download_weights.sh x --source hf`;
- **прокси:** `HTTPS_PROXY=http://proxy:port bash scripts/download_weights.sh`;
- **копия с другой машины:** `scp weights/*.pth user@server:<repo>/weights/` и `scp ckpts/*.pth user@server:<repo>/ckpts/`. Уже лежащие файлы скрипт не перекачивает.

## 2. Конфигурация: только `experiment.yml`

Редактируется один файл. Основные поля:

| Поле | Что задаёт |
|---|---|
| `name`, `output_root` | папка результатов `outputs/<name>/` |
| `model` | `s` \| `m` \| `l` \| `x` |
| `weights` | `auto` (скачанный COCO-чекпоинт), путь к чекпоинту или `best.pth`, `none` (только backbone) |
| `data` | `format: csv` (путь к CSV, `class_merge`) или `format: coco` (пути к изображениям и json) |
| `input_size` | `[h, w]`, кратно 32. Прямоугольник (например `[128, 800]` для полос 1600×256) автоматически выключает Mosaic |
| `epochs`, `batch_size`, `workers` | расписание рецепта модели пересчитывается под `epochs` |
| `devices` | номера GPU, как в `nvidia-smi`. Несколько номеров включают DDP, см. [Несколько GPU](#несколько-gpu-ddp) |
| `optimizer`, `lr`, `backbone_lr_ratio`, `weight_decay` | `sgd` или `adamw`; `auto` — значения рецепта (AdamW) или 0.01 / ×0.02 / 1e-4 (SGD) |
| `best_metric` | `map50` или `map`: по этой метрике выбирается `best.pth` |

### Данные в CSV

Одна строка на бокс, лишние столбцы игнорируются:

| Столбец | Значение |
|---|---|
| `image_path` | абсолютный путь к изображению (строка) |
| `instance_label` | класс (строка) |
| `bbox_x_tl`, `bbox_y_tl`, `bbox_x_br`, `bbox_y_br` | левый верхний и правый нижний угол, пиксели исходного изображения |
| `split` | `train`, `val` или `test`. Строки с другими значениями пропускаются с предупреждением |

- Пустые `instance_label` или bbox означают изображение без боксов: чистое изображение или неразмеченный test.
- `class_merge` объединяет классы по строковым меткам, например `{scratch_small: scratch}`. Классы нумеруются 0..K-1 в отсортированном порядке, одинаково для всех выборок.
- CSV конвертируется в COCO один раз, в `data_cache/<name>/`. Изображения не копируются. При изменении CSV конвертация повторяется.
- Старый формат `train_bboxes.csv` (`ImageId, ClassId, x_min, y_min, x_max, y_max, split` плюс папка `images`) определяется по заголовку.

`tools/steel/prepare_experiment.py` собирает из этого файла полный конфиг DEIMv2
`configs/_generated/<name>.yml`. Его не нужно править руками: он перезаписывается при каждом запуске.

## 3. Обучение

```bash
bash train.sh                     # = python tools/steel/prepare_experiment.py experiment.yml --train
bash train.sh my_experiment.yml   # другой файл эксперимента
```

Результаты лежат в `outputs/<name>/`:
- `best.pth` — только лучшие веса (EMA) по `best_metric`, вместе с конфигом модели и именами классов: для инференса больше ничего не нужно. Промежуточные чекпоинты DEIM ведёт в `outputs/<name>/.state/` и удаляет после обучения;
- `metrics.png` и `metrics.csv` — обновляются после каждой эпохи: mAP@0.5, mAP@0.5:0.95, precision, recall, F1 (точка лучшего F1 при IoU 0.5), AP@0.5 по классам, train loss, lr;
- `log.txt`, `train.log`, `summary/` (TensorBoard).

Дообучить ещё раз с лучших весов: `weights: outputs/<name>/best.pth`.

### Несколько GPU (DDP)

```yaml
devices: [0, 1, 2, 3]   # в experiment.yml; запуск тот же: bash train.sh
```

При нескольких номерах `train.sh` запускает `torchrun --standalone --nproc_per_node=N train.py ...`: это штатный DDP
DEIM с SyncBatchNorm, свободный порт выбирается сам. `batch_size` и `val_batch_size` задают общий batch на все
GPU, как `total_batch_size` в DEIM: на каждой карте `batch_size / N`, число итераций за эпоху, lr и warmup те же,
что на одной карте. Поэтому batch должен делиться на N, а чтобы занять память каждой карты, его увеличивают в N раз.
`workers` считаются на каждый процесс. Лучшие веса, метрики и графики пишет только процесс rank 0.
DDP работает только на Linux. `torchrun` из Windows-сборок torch 2.7 не запускается (они собраны без libuv),
поэтому `train.sh` на Windows с несколькими `devices` остановится с сообщением об этом.

## 4. Оценка

```bash
python tools/steel/evaluate.py experiment.yml --fps                  # val: mAP, P/R/F1, AP по классам, FPS
python tools/steel/evaluate.py experiment.yml --split test --conf 0.3
```

В `outputs/<name>/` появляются:
- `<split>_predictions.csv` — предсказания в формате инференса (ниже), `confidence ≥ --conf`;
- `<split>_predictions.json` — все боксы в формате COCO;
- `eval_<split>.json` — метрики, если у выборки есть разметка. Для неразмеченного test создаются только предсказания.

FPS меряется при batch 1 и включает чтение, модель и постобработку.

## 5. Инференс

```python
from tools.steel.infer import Detector   # из корня репозитория; иначе сначала sys.path.insert(0, "<путь к deim-steel>")

det = Detector("outputs/deimv2_x_800x128_sgd/best.pth")        # device="cuda" по умолчанию, если есть GPU
df = det.predict("/data/new_images", conf=0.3)                 # папка
df = det.predict(["/data/a.jpg", "/data/b.jpg"])               # список путей
df = det.predict(my_df)                                        # DataFrame или CSV со столбцом image_path (+ split)
```

То же из командной строки (результат по умолчанию пишется в `predictions.csv` рядом с весами):

```bash
python tools/steel/infer.py outputs/deimv2_x_800x128_sgd/best.pth /data/new_images --conf 0.3 --out preds.csv
```

`predict` возвращает pandas DataFrame в формате обучающего CSV плюс `confidence`, одна строка на бокс:

| Столбец | Значение |
|---|---|
| `image_path` | путь как на входе; относительный путь превращается в абсолютный |
| `instance_label` | имя класса, как в обучающих данных |
| `bbox_x_tl`, `bbox_y_tl`, `bbox_x_br`, `bbox_y_br` | углы бокса в пикселях исходного изображения, обрезаны по его границам |
| `split` | из входного CSV или DataFrame, иначе аргумент `split` (по умолчанию `test`) |
| `confidence` | уверенность модели, 0..1. В строке остаются боксы с `confidence ≥ conf` |

Изображение без боксов выше порога даёт одну строку с пустыми `instance_label`, bbox и `confidence`, как в
обучающем CSV (`keep_empty=False` или `--no-empty` такие строки убирает). Поэтому результат можно без изменений
подать обратно в `csv_to_coco`, например как псевдоразметку. Сырые результаты без порога и имён классов отдаёт
`det.detect(paths)`: по изображению выдаются `(path, (w, h), labels, boxes_xyxy, scores)`.

По умолчанию модель собирается в режиме deploy, как в upstream `tools/inference`: conv+BN слиты, декодер обрезан
до `eval_idx`. Так быстрее, но уверенности отличаются от валидации при обучении в 3–4-м знаке. `Detector(...,
deploy=False)` воспроизводит валидацию точно. Этим режимом пользуется `evaluate.py`, поэтому его mAP совпадает с
`metrics.csv`.

Чекпоинтам без встроенного конфига (сохранённым до этой версии или `last.pth`/`best_stg*.pth` из upstream DEIM)
нужен конфиг обучения: `Detector(ckpt, config="configs/_generated/<name>.yml")`. Для `outputs/<name>/` он находится сам.

## Старый способ

Upstream-запуск работает без изменений. Новые возможности включаются только ключами, которые пишет генератор
(`save_best_only`, `best_metric`, `plot_metrics`):

```bash
python train.py -c configs/deimv2/deimv2_dinov3_x_coco.yml --use-amp --seed=0 -t weights/deimv2_dinov3_x_coco.pth
```

## Изменения относительно upstream

| Файл | Зачем |
|---|---|
| `engine/data/transforms/_transforms.py` | совместимость с torchvision ≥ 0.21 (`_transform` → `transform`) |
| `engine/misc/profiler_utils.py` | подсчёт FLOPs на `eval_spatial_size`: прямоугольный вход падал на квадратной заглушке |
| `engine/optim/optim.py` | `SGDIgnoreBetas`: SGD, которому не мешает `betas` из базовых AdamW-конфигов |
| `engine/solver/det_solver.py` | `save_best_only`, `best_metric`, `plot_metrics` (по умолчанию выключены); `best.pth` хранит конфиг и имена классов |
| `engine/misc/dist_utils.py` | gloo, если нет NCCL; при `WORLD_SIZE > 1` ошибка инициализации DDP больше не превращается молча в N независимых обучений |
| `engine/core/yaml_utils.py` | `load_config` без общего словаря по умолчанию: второй конфиг в том же процессе смешивался с первым |
| `engine/misc/metrics_log.py` | `metrics.csv` и `metrics.png` по эпохам |
| `tools/steel/*`, `scripts/download_weights.sh`, `train.sh`, `experiment.yml` | единый конфиг, запуск DDP, загрузка весов, конвертация CSV → COCO и DINOv3 из timm, оценка, инференс |

## Замечания

- **Windows: при нехватке видеопамяти драйвер молча переносит её в системную RAM вместо ошибки OOM**, и обучение замедляется в десятки раз. Batch подбирайте с запасом. Для X на 128×800 и 16 ГБ: batch 20 занимает около 12 ГБ, batch 24 уже переполняет.
- **Прямоугольный вход:** Mosaic и multi-scale в DEIM строят квадрат, поэтому для прямоугольника они выключаются. Проверка в `prepare_experiment.py` не даст включить их вручную.
- **Меньше 3 эпох:** финальная стадия без аугментаций не создаётся, иначе смена стадий попала бы на эпоху 0.
