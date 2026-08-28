# Kernelyra — внутренний полный справочник пользователя и интегратора

Статус: **0.6.0a2 (alpha)**. Это расширенная внутренняя справка для работы с
исходным checkout: она намеренно **не включается** в wheel, sdist, source ZIP
или GitHub Release assets. Публичные README остаются краткими и не ссылаются
на этот файл, поэтому архив релиза остаётся самодостаточным.

## 1. Назначение и честные границы

Kernelyra — локальная terminal-first библиотека для tabular training.

Поддерживаемые задачи:
  - binary_classification;
  - multiclass_classification;
  - regression.

Поддерживаемые источники обучения:
  - CSV, TSV, JSONL и NDJSON;
  - numeric NPZ;
  - Parquet/PQ после установки extra [data].

Реально обучаемые backend/architecture:
  - native + linear;
  - numpy + linear;
  - torch + mlp, если установлен extra [torch];
  - tensorflow + mlp, если установлен extra [tensorflow].

Kernelyra сейчас НЕ включает встроенный trainer для LLM, изображений, аудио,
видео, 3D, graph neural network, CNN или transformer. Название, которое CLI
может распознать для диагностики, не является доказательством обучения или
поддержки model container.

Проверяемая платформа: Windows x64, Python 3.11–3.13.


2. УСТАНОВКА
-------------

PowerShell:

  git clone https://github.com/Solveran601/Kernelyra.git
  Set-Location Kernelyra
  py -3.13 -m venv .venv
  .\.venv\Scripts\python -m pip install -e .

Дополнительные возможности из того же checkout:

  .\.venv\Scripts\python -m pip install -e ".[data]"
  .\.venv\Scripts\python -m pip install -e ".[parquet]"
  .\.venv\Scripts\python -m pip install -e ".[torch]"
  .\.venv\Scripts\python -m pip install -e ".[tensorflow]"
  .\.venv\Scripts\python -m pip install -e ".[full]"
  .\.venv\Scripts\python -m pip install -e ".[gateway]"
  .\.venv\Scripts\python -m pip install -e ".[mcp]"

Проверка установки:

  .\.venv\Scripts\kernelyra version
  .\.venv\Scripts\kernelyra capabilities
  .\.venv\Scripts\kernelyra formats
  .\.venv\Scripts\kernelyra native status


3. ЛОГИКА РАБОТЫ
----------------

  файл/папка
    -> bounded router inspection
    -> Data Doctor и dataset contract
    -> TrainingPlan
    -> import_file (memory) ИЛИ attach_path (stream)
    -> backend training
    -> checkpoints + Model Guard
    -> held-out evaluation + report

Doctor описывает только ограниченный preview. Для CSV/TSV/JSONL/NDJSON он и
планировщик используют одну оценку количества строк с чтением не более 8 MiB.

Правило split зависит от выбранного пути данных:
  - явный group_column на streaming path: context split; все строки одной
    группы остаются в одном split, а group-колонка исключается из признаков;
  - автоматически найденная group/user/account/customer/session/conversation/
    document/device колонка на streaming path получает ту же защиту;
  - streaming без context: stable_record_hash, то есть целая строка
    детерминированно назначается train/validation/test;
  - materialized classification: deterministic stratified split;
  - materialized time-like input: temporal, исходный порядок не меняется;
  - остальные materialized случаи: deterministic random split.

В streaming должен получиться минимум 8 строк и в validation, и в test. Если
малое число групп или доли split этого не дают, библиотека прерывает preflight
с объяснением до старта worker, а не создаёт неполноценное обучение.

Пак алгоритмов меняет потоки, bulk-вызовы, prefetch, чанки и размер арены. Он
не классифицирует компьютер пользователя и не отменяет явные лимиты.


4. САМЫЙ КОРОТКИЙ PYTHON-ПУТЬ
-----------------------------

  from kernelyra import Engine

  with Engine("./project") as k:
      health = k.doctor("train.csv", "label")
      plan = k.plan("train.csv", "label", backend="native", execution="cpu")
      result = k.fit("train.csv", "label", backend="native", execution="cpu")
      report = k.report(result.run.id, "./project/report.json")

  print(plan.batch_size)
  print(report["output"])

Планируй перед длинным запуском. TrainingPlan содержит backend, batch_size,
data_mode, лимиты, sources каждого значения, warnings, data contract и split
policy.


5. TRAININGCONFIG — ПОЛНЫЙ FLUENT-СИНТАКСИС
-------------------------------------------

Создание и композиция:

  from kernelyra import TrainingConfig, Config, Settings

  config = TrainingConfig()
  config.set(target="label", backend="native")
  config = TrainingConfig.from_mapping({"execution": "cpu"})
  copied = config.copy()
  config.merge({"max_steps": 500}, TrainingConfig().seed(7))
  config.unset("max_steps", "batch_size")
  config.automatic()              # сбросить всё к automatic
  config.automatic("cpu", "ram") # сбросить только указанные поля
  payload = config.to_dict()

Config и Settings — псевдонимы TrainingConfig.

Задача, backend и модель:

  config.target("label")
  config.task("binary_classification")
  config.backend("native")
  config.architecture("linear")
  config.model_format("kernelyra-npz")
  config.execution("cpu")      # auto | cpu | hybrid
  config.cpu_only()
  config.hybrid()
  config.pack("throughput")    # careful | balanced | throughput | maximum | custom name

profile(), low_memory(), weak(), balanced(), medium(), performance(),
powerful(), workstation(), custom() и hardware() оставлены для совместимости.
В новом коде используй execution(), pack() и budget().

Лимиты:

  config.resources(cpu=70, ram=55, gpu=0, threads=6)
  config.budget(
      cpu_percent=70,
      memory_percent=55,
      gpu_percent=0,
      threads=6,
  )

cpu, ram и gpu — проценты от обнаруженного ресурса. CPU допустим 10..100,
RAM 10..95, GPU 0..100. threads не может быть больше обнаруженных CPU threads.
Для execution="cpu" gpu обязан быть 0. Неуказанный аргумент остаётся automatic.

Данные, скорость и память:

  config.data(mode="auto")
  config.data(mode="memory", workers=0, prefetch=1)
  config.data(mode="stream", workers=2, prefetch=2)

mode:
  auto   — безопасная автоматическая рекомендация;
  memory — наиболее быстрый путь для набора, помещающегося в лимит памяти;
  stream — внешний потоковый путь с ограниченным RAM и безопасным context split
           на CSV/TSV/JSONL/NDJSON/Parquet.

Принудительный memory отклоняется для слишком большого файла и context split.
Принудительный stream отклоняется, если у формата нет streaming reader.
workers: 0..64; prefetch: 0..32.

Split, контекст и неравномерные чанки:

  config.split(
      validation_percent=20,
      test_percent=10,
      group_column="customer_id",
  )
  config.chunks(
      target_records=4096,
      minimum_records=2048,
      maximum_records=6144,
  )

validation_percent и test_percent допустимы от 0 до 95; вместе они не могут
превысить 95, потому что минимум 5% остаётся train. `seed(42)` сохраняет
историческое context-назначение; другое значение seed воспроизводимо меняет
назначение групп. `chunks()` создаёт непрерывные неравные диапазоны и требует
`minimum ≤ target ≤ maximum`; финальный чанк может быть меньше minimum, чтобы
точно завершить датасет. В summary поля `configured_minimum_records` и
`configured_maximum_records` показывают введённые границы, а
`minimum_records`/`maximum_records` — фактически получившиеся размеры.

Обучение и контроль качества:

  config.goal(0.92)
  config.steps(1_400)
  config.batch()                         # вернуть batch в automatic
  config.batch(128, accept_risk=True)
  config.optimizer(learning_rate=0.001, weight_decay=0.0001)
  config.model(128, 64, precision="float32")
  config.stopping(
      maximum_steps=5_000,
      target_metric=0.94,
      early_stopping_patience=30,
      target_patience=3,
  )
  config.quality(
      evaluation_interval=25,
      min_improvement=0.001,
      early_stopping_patience=30,
      target_patience=3,
  )
  config.guard(margin=0.03, patience=3)
  config.checkpoints(
      resume="last",   # last | best; "last" по умолчанию
      final="best",    # best | last; checkpoint для итоговой проверки и результата
      rollback="best", # единственный безопасный вариант для Model Guard
  )
  config.seed(42)

max_steps: 1..10_000_000. hidden_layers: 1..16 положительных ширин.
precision: auto, float16, bfloat16, float32, float64. Ручной batch за
безопасным диапазоном требует accept_risk=True. guard() не выключает Model
Guard, а меняет его чувствительность.

Checkpoint-политика разделяет три разные задачи: resume продолжает последний
сохранённый шаг (и состояние оптимизатора для PyTorch/TensorFlow), final
выбирает checkpoint для финальной held-out проверки и результата, rollback
остаётся best. Нельзя переключить rollback на last: это намеренная защита от
закрепления ухудшившейся модели.


6. ENGINE И ВЕРХНЕУРОВНЕВЫЙ API
--------------------------------

  Engine(workspace=".", *, config=None, settings=None)
  engine.configure(settings=None, **options)
  engine.hardware
  engine.capabilities
  engine.inspect(dataset)
  engine.doctor(dataset, target=None)
  engine.plan(dataset, target=None, *, settings=None, **options)
  engine.fit(dataset, target=None, *, settings=None, **options)
  engine.train(...)               # alias fit
  engine.finetune(model, dataset, target=None, *, settings=None, **options)
  engine.plan_many(datasets, target=None, *, settings=None, **options)
  engine.fit_many(datasets, target=None, *, settings=None, **options)
  engine.report(run_id, output=None)
  engine.close()

Engine — context manager. fit_many запускает задачи последовательно, чтобы
ресурсные ограничения не обходились параллельными работами.

Одноразовые функции:

  from kernelyra import plan, train, finetune, fit

  plan(dataset, workspace=".", config=None, **options)
  train(dataset, workspace=".", config=None, **options)
  finetune(model, dataset, workspace=".", config=None, **options)
  fit(dataset, target=None, workspace=".", config=None, settings=None, **options)

Приоритет конфигурации:
  явный аргумент -> KERNELYRA_* environment -> kernelyra.toml -> automatic.


7. ПОЛНЫЙ СПИСОК ПАРАМЕТРОВ PLAN/TRAIN/FINETUNE
------------------------------------------------

target, task, backend, architecture, model_format, profile, execution,
algorithm_pack, batch_size, accept_batch_risk, max_steps, target_metric,
learning_rate, weight_decay, hidden_layers, precision, data_mode, cpu, ram,
gpu, threads, data_workers, prefetch, seed, evaluation_interval,
min_improvement, degradation_margin, degradation_patience,
early_stopping_patience, target_patience, checkpoint_resume, checkpoint_final,
checkpoint_rollback, validation_percent, test_percent, group_column,
chunk_target_records, chunk_minimum_records, chunk_maximum_records, name.

task: auto | binary_classification | multiclass_classification | regression.
backend: auto | native | numpy | torch | tensorflow.
execution: auto | cpu | hybrid.
data_mode: auto | memory | stream.
precision: auto | float64 | float32 | float16 | bfloat16.

Переменные окружения:

  KERNELYRA_TARGET              KERNELYRA_TASK
  KERNELYRA_BACKEND             KERNELYRA_ARCHITECTURE
  KERNELYRA_MODEL_FORMAT        KERNELYRA_EXECUTION
  KERNELYRA_ALGORITHM_PACK      KERNELYRA_BATCH_SIZE
  KERNELYRA_MAX_STEPS           KERNELYRA_TARGET_METRIC
  KERNELYRA_CPU_PERCENT         KERNELYRA_RAM_PERCENT
  KERNELYRA_GPU_PERCENT         KERNELYRA_THREADS
  KERNELYRA_SEED                KERNELYRA_LEARNING_RATE
  KERNELYRA_WEIGHT_DECAY        KERNELYRA_HIDDEN_LAYERS
  KERNELYRA_PRECISION           KERNELYRA_DATA_MODE
  KERNELYRA_DATA_WORKERS        KERNELYRA_PREFETCH
  KERNELYRA_EVALUATION_INTERVAL KERNELYRA_MIN_IMPROVEMENT
  KERNELYRA_DEGRADATION_MARGIN  KERNELYRA_DEGRADATION_PATIENCE
  KERNELYRA_EARLY_STOPPING_PATIENCE KERNELYRA_TARGET_PATIENCE
  KERNELYRA_CHECKPOINT_RESUME   KERNELYRA_CHECKPOINT_FINAL
  KERNELYRA_CHECKPOINT_ROLLBACK
  KERNELYRA_VALIDATION_PERCENT  KERNELYRA_TEST_PERCENT
  KERNELYRA_GROUP_COLUMN
  KERNELYRA_CHUNK_TARGET_RECORDS KERNELYRA_CHUNK_MINIMUM_RECORDS
  KERNELYRA_CHUNK_MAXIMUM_RECORDS

Пример kernelyra.toml:

  [training]
  execution = "cpu"
  algorithm_pack = "throughput"
  backend = "native"
  data_mode = "memory"
  cpu = 85
  ram = 65
  threads = 8
  max_steps = 2000
  evaluation_interval = 25
  early_stopping_patience = 30
  validation_percent = 20
  test_percent = 10
  group_column = "customer_id"
  chunk_target_records = 4096
  chunk_minimum_records = 2048
  chunk_maximum_records = 6144


8. CLI: ПОЛНАЯ КАРТА
--------------------

Все актуальные параметры: kernelyra --help и kernelyra plan --help.
Стабильный JSON: kernelyra --json <command>.
Общие флаги: --workspace PATH, --json, --timeout SECONDS, --daemon-url URL,
--autostart.

Диагностика и план:

  kernelyra version
  kernelyra doctor
  kernelyra capabilities
  kernelyra formats
  kernelyra advise PATH
  kernelyra execution
  kernelyra tune --execution cpu --pack throughput --records 100000 --features 64 --batch-size 128
  kernelyra chunk-plan 100000 --target-records 4096 --minimum-records 2048 --maximum-records 6144 --seed 42 --validation-percent 20 --test-percent 10
  kernelyra plan DATASET --target label --backend native --execution cpu --data-mode memory
  kernelyra train DATASET --target label --backend native --execution cpu --data-mode stream --validation-percent 20 --test-percent 10 --group-column customer_id --chunk-target-records 4096 --chunk-minimum-records 2048 --chunk-maximum-records 6144
  kernelyra finetune MODEL DATASET --target label --backend torch

Data Doctor и датасеты:

  kernelyra dataset inspect PATH
  kernelyra dataset doctor PATH --target label
  kernelyra dataset import PATH --target label
  kernelyra dataset add PATH --target label
  kernelyra dataset list --limit 100 --offset 0
  kernelyra dataset show DATASET_ID
  kernelyra dataset remove DATASET_ID

Паки:

  kernelyra packs list
  kernelyra packs path
  kernelyra packs algorithms
  kernelyra packs show NAME
  kernelyra packs clone my-pack --from balanced --label "My pack"
  kernelyra packs add-algorithm my-pack thread_parallel_gradient
  kernelyra packs remove-algorithm my-pack thread_parallel_gradient
  kernelyra packs delete my-pack

Встроенные паки immutable. Редактировать можно только clone.

Запуски, результаты и обслуживание:

  kernelyra daemon start | foreground | status | stop
  kernelyra run create --dataset DATASET_ID [--start]
  kernelyra run start|pause|resume|stop RUN_ID
  kernelyra run show|get|watch RUN_ID
  kernelyra run logs RUN_ID --limit 100
  kernelyra run trace RUN_ID
  kernelyra run export RUN_ID --output PATH
  kernelyra run list --status STATUS --limit 100 --offset 0
  kernelyra infer RUN_ID --requests 200
  kernelyra report RUN_ID --output report.json
  kernelyra native status
  kernelyra native build --output PATH
  kernelyra config validate PATH
  kernelyra migrate
  kernelyra repair [--apply]
  kernelyra cleanup [--apply]
  kernelyra workspace export-manifest PATH
  kernelyra workspace import-manifest PATH
  kernelyra rpc [--config kernelyra.toml]
  kernelyra mcp [--config kernelyra.toml]


8.1 NATIVE-ЯДРА И ТРАССА ВЫПОЛНЕНИЯ 0.6
-----------------------------------------

У native backend пять разных ролей, а не пять независимых библиотек, которые
нужно вручную запускать:

  c-abi             — стабильная граница C ABI и C-core, который до выделения
                      памяти проверяет размеры таблицы, minibatch и переполнения;
  cpp-dispatch      — выбор SIMD/OpenMP и безопасного fallback;
  rust-policy       — детерминированные split/chunk policies, batch plan и
                      random sampler без Python-выделения индексов;
  fortran-numeric   — переносимые dense gradient/update kernels с preflight:
                      следующий float32 weight/bias проверяется до записи;
  zig-memory        — bounded memory, preprocessing и единый batch gather
                      признаков вместе с matching target.

`kernelyra native status` сообщает, что DLL собрана с компонентами и какие
компоненты включены. Это НЕ является доказательством их использования в каждом
вызове. В 0.6 у NativeModel есть накопительная фактическая трасса:

  from kernelyra.native_core import NativeCore, NativeModel
  import numpy as np

  core = NativeCore()
  x = np.ascontiguousarray(np.random.default_rng(7).normal(size=(128, 8)), dtype=np.float32)
  y = np.ascontiguousarray((x[:, 0] > 0).astype(np.float32))
  with NativeModel(task="binary_classification", features=8, threads=1, core=core) as model:
      loss = model.train_random_step(x, y, batch_size=32)
      print(model.execution_trace())

При полном включённом core этот короткий random-batch путь наблюдает mask 31 и
все пять имён. Его порядок: C проверяет контракт размерностей → Rust выбирает
индексы → Zig переносит строки и targets → Fortran выполняет guarded update.
Большой full-batch путь может намеренно выбрать C++ AVX2/FMA и не вызывать
Rust/Zig sampler/gather; это нормальная оптимизация, а не ошибка.
В AutoTrainer последняя трасса сохраняется в session.metadata["native_execution"].

Fused preprocessing доступен через NativeCore.preprocess_f32(...): при
одновременных impute + normalize + clip Zig выполняет совместимую float32
операцию за один проход. При других комбинациях флагов остаётся проверяемый
fallback. Эта деталь не расширяет список поддерживаемых форматов или моделей.

В 0.6.0a2 Rust policy получил ABI-совместимый seeded context split:
`kr_rust_split_for_key_seeded(group_key, seed, validation_percent, test_percent)`.
Через C доступен `kr_c_context_split_seeded(...)`, а Python вызывает его через
`NativeCore.split_for_context(..., seed=...)`. Старый ABI не удалён: старые DLL
работают с совместимым fallback, а новый DLL использует Rust непосредственно.


9. POWERSHELL
-------------

Подключение:

  Import-Module .\powershell\Kernelyra.psd1 -Force
  Get-Command -Module Kernelyra

Основные команды:

  New-KernelyraProject -Path .\project
  Test-KernelyraDataset .\data\train.csv -Target label
  Get-KernelyraDataContract .\data\train.csv -Target label
  Get-KernelyraPlan .\data\train.csv -Target label -Execution cpu -Pack throughput -DataMode memory -Cpu 85 -Ram 65 -Threads 8
  Start-KernelyraTraining .\data\train.csv -Target label -Execution cpu -Pack throughput -DataMode stream -MaxSteps 2000 -ValidationPercent 20 -TestPercent 10 -GroupColumn customer_id -ChunkTargetRecords 4096 -ChunkMinimumRecords 2048 -ChunkMaximumRecords 6144
  Watch-KernelyraRun RUN_ID
  Resume-KernelyraRun RUN_ID
  Get-KernelyraRunStatus RUN_ID
  Export-KernelyraModel RUN_ID -Output .\model.zip
  Get-KernelyraReport RUN_ID -Output .\report.json
  Get-KernelyraExecution
  Get-KernelyraNativeStatus
  Get-KernelyraCpuTuning -Records 100000 -Features 64 -BatchSize 128
  Get-KernelyraChunkPlan -Records 100000 -TargetRecords 4096 -MinimumRecords 2048 -MaximumRecords 6144 -ValidationPercent 20 -TestPercent 10

Паки PowerShell:

  Get-KernelyraPack
  Get-KernelyraPackAlgorithm
  Get-KernelyraPackTablePath
  Copy-KernelyraPack my-pack -Base balanced
  Add-KernelyraPackAlgorithm my-pack thread_parallel_gradient
  Remove-KernelyraPackAlgorithm my-pack thread_parallel_gradient
  Remove-KernelyraPack my-pack
  Invoke-Kernelyra

New-KernelyraProject и Start-KernelyraTraining поддерживают WhatIf/Confirm.


10. JSONL ПРОТОКОЛ ДЛЯ SDK
--------------------------

Запуск: kernelyra rpc
Первая строка: {"type":"ready","protocol":"kernelyra-jsonl/1"}
Одна UTF-8 JSON-строка запроса ограничена 1 MiB.

  {"id":"p1","method":"ping","params":{}}
  {"id":"p2","method":"capabilities","params":{}}
  {"id":"p3","method":"hardware","params":{}}
  {"id":"p4","method":"plan","params":{"dataset":"train.csv","target":"label","execution":"cpu"}}
  {"id":"p5","method":"train","params":{"dataset":"train.csv","target":"label","backend":"native"}}
  {"id":"p6","method":"finetune","params":{"model":"model.npz","dataset":"train.csv","backend":"numpy"}}

Ответ успеха: {"id":"p1","ok":true,"result":{...}}
Ошибка:        {"id":"p1","ok":false,"error":"...","error_type":"ConfigurationError"}

Методы: ping, capabilities, hardware, plan, train, finetune. Параметры
plan/train/finetune совпадают с разделом 7.


11. ДРУГИЕ ПУБЛИЧНЫЕ PYTHON-ИНСТРУМЕНТЫ
----------------------------------------

| API | Назначение |
|---|---|
| `analyze_inspection`, `inspect_path`, `recommend_chunk_policy` | bounded Data Doctor, signed contract и preview чанков |
| `ContextChunkPlanner`, `ContextChunk` | детерминированные context split, `split_for`, `split_indices`, `partition`, `chunk_ranges`, `summary` |
| `autotune_execution` | только расчёт tuning; обучение и скрытый benchmark не запускаются |
| `list_algorithm_packs`, `get_algorithm_pack`, `algorithm_pack_table`, `algorithm_pack_path` | чтение built-in/custom pack и пути JSON custom packs |
| `create_algorithm_pack`, `add_pack_algorithm`, `remove_pack_algorithm`, `delete_algorithm_pack` | валидируемое создание и изменение только custom pack |
| `build_experiment_report`, `write_experiment_report` | собрать/записать JSON или HTML отчёт completed run |
| `run_inference_check` | held-out inference; хеш checkpoint до/после доказывает отсутствие мутации |
| `extract_text`, `extract_folder`, `text_format_count` | доступное извлечение и статистика текстовых форматов |
| `ByteTokenizer`, `plan_text_for_training`, `prepare_masked_text_examples`, `batch_masked_text_examples`, `iter_masked_text_batches` | UTF-8 byte preparation и causal masks; это не LLM trainer |
| `Workspace`, `Kernelyra`, `RunHandle` | низкоуровневый локальный workspace и управление run |
| `DaemonClient`, `AsyncKernelyraClient`, `KernelyraClient` | sync/async JSONL clients; последний — alias sync-клиента |
| `NativeTensorArena` | opt-in bounded aligned float32 buffers; обычный `Engine` сам его не требует |
| `QualityGate` | finite metrics и gap к best/baseline до замены безопасного checkpoint |
| `Config`, `Settings`, `Dataset`, `Run` | aliases `TrainingConfig`, `DatasetInfo`, `RunInfo` |
| `BackendInfo`, `DatasetManifest`, `DatasetSchema`, `IngestorInfo`, `RunConfig`, `RunMetrics`, `RunStatus`, `TaskType` | immutable metadata-типы публичного API |
| `KernelyraError` и наследники | единая и конкретная обработка configuration/dataset/run/worker/access ошибок |


12. БЕЗОПАСНЫЕ РЕЦЕПТЫ
-----------------------

Быстрый CPU, если данные безопасно помещаются в RAM:

  TrainingConfig().execution("cpu").pack("throughput").data(
      mode="memory"
  ).budget(cpu_percent=100, memory_percent=85, threads=12)

Экономный RAM и защита контекста:

  TrainingConfig().execution("cpu").pack("careful").data(
      mode="stream", workers=1, prefetch=1
  ).budget(cpu_percent=60, memory_percent=40, threads=4)

Без ручных решений:

  TrainingConfig().automatic()

Не переносить локальный benchmark на другое железо или задачу. Утверждение о
скорости требует версии backend, датасета, команд и измерения на том же CPU/GPU.


13. ЧАСТЫЕ ОШИБКИ
-----------------

"format is recognized but no trainable ingestor is installed"
  Формат определён, но обучающего adapter нет.

"data_mode='memory' is unsafe"
  Датасет не проходит лимит RAM либо требует context-safe streaming.

"has no streaming reader"
  Выбран stream для формата без streaming path.

"hybrid execution requires a detected accelerator"
  Выбран hybrid без доступного accelerator.

"Batch ... exceeds safe range"
  Ручной batch не прошёл защиту; оставь auto или передай accept_risk=True.

"Unknown training option"
  Использовано неверное или внутреннее имя; см. полный список в разделе 7.

Конец единого справочника.
