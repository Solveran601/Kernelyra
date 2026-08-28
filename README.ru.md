<p align="center">
  <img src="assets/brand/kernelyra-logo.png" alt="Kernelyra" width="520">
</p>

<p align="center">
  <a href="README.md">English version</a>
</p>

<p align="center"><img src="assets/brand/kernelyra-mark-animated.svg" alt="Анимированный знак Kernelyra" width="72"></p>

<p align="center"><strong>Нативное обучение табличных данных с контролем ресурсов.</strong></p>

Kernelyra **0.6.0a1 (alpha)** — локальная библиотека для обучения табличных
моделей из терминала. Один и тот же путь планирования доступен в CLI, Python
API, PowerShell-модуле и JSONL-протоколе для поставляемых SDK.

<p align="center">
  <a href="#capabilities">Возможности</a> ·
  <a href="#install">Установка</a> ·
  <a href="#powershell">PowerShell</a> ·
  <a href="#limits">Ограничения</a>
</p>

<a id="capabilities"></a>

## Что работает в этой alpha-версии

- Бинарная и многоклассовая классификация, а также регрессия на табличных
  данных.
- Прямое обучение на CSV, TSV, JSONL/NDJSON, числовых NPZ и опциональном
  Parquet.
- Встроенные native и NumPy backend; PyTorch и TensorFlow/Keras — опционально,
  когда они установлены. В native-core есть пять наблюдаемых ролей: C ABI,
  C++ dispatcher, Rust policy, Fortran numeric kernels и Zig memory kernels.
  Поле `native_execution` показывает только движки, которые действительно
  участвовали в вызовах конкретной модели; оно не считает все пять «по умолчанию».
- Явный выбор выполнения `cpu` или `hybrid`, заданные разработчиком лимиты
  CPU/RAM/GPU/потоков, четыре встроенных пакета алгоритмов (`careful`,
  `balanced`, `throughput`, `maximum`) и проверяемые пользовательские паки на
  их основе. Пакеты реально меняют потоки, bulk-вызовы, чанки, предзагрузку и
  размер арены, но не классифицируют компьютер пользователя. По умолчанию
  checkpoint-resume идёт с `last`; для итоговой проверки можно выбрать `best`
  или `last`, а Model Guard восстанавливает только `best`.
- Data Doctor: ограниченная предварительная проверка, подписанный контракт
  датасета, детерминированная рекомендация split и план неравномерных чанков.
- Подготовка UTF-8-текста: native-планирование чанков при доступном native-core,
  обратимый byte-токенизатор и causal loss-mask для будущего тренера. Это не
  LLM-тренер.
- Model Guard V2: проверка конечности метрик и сохранение тренда качества в
  health-записи run.
- Переносимые JSON/HTML-отчёты эксперимента и явные лимиты CPU/RAM/GPU.

Data Doctor намеренно ограничен выборкой: выводы относятся к проверенным
строкам, а не ко всему датасету. Для materialized training классификация
разделяется детерминированно по классам, а time-like колонка сохраняет исходный
порядок входных строк. Для CSV, TSV, JSONL/NDJSON и Parquet AutoTrainer при
обнаружении group/context-колонки переводит данные в streaming split: каждый
контекст остаётся в одном split, а его идентификатор исключается из признаков.
Для форматов без такого streaming-пути остаётся явное предупреждение об утечке.

<a id="install"></a>

## Установка из исходников

Публикации на PyPI и готовых GitHub-релизов пока нет. Устанавливай исходную
копию с Python 3.11–3.13.

```powershell
git clone https://github.com/Solveran601/Kernelyra.git
Set-Location Kernelyra
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
```

`.[data]` нужен для Parquet, `.[torch]` — для PyTorch, а `.[tensorflow]` — для
TensorFlow/Keras.

<a id="powershell"></a>

## CLI из PowerShell

```powershell
python -m kernelyra version
python -m kernelyra doctor
python -m kernelyra execution
python -m kernelyra plan .\data\train.csv --target label --execution cpu --pack throughput --cpu 100 --ram 85 --threads 12
```

При необходимости сначала выполни явную проверку данных:

```powershell
python -m kernelyra dataset doctor .\data\train.csv --target label
```

Посмотри таблицу паков и доступные алгоритмы, затем создай изменяемый пак, не
трогая встроенный эталон:

```powershell
python -m kernelyra packs list
python -m kernelyra packs algorithms
python -m kernelyra packs clone my-careful --from careful
python -m kernelyra packs add-algorithm my-careful thread_parallel_gradient
python -m kernelyra tune --execution cpu --pack my-careful --records 100000 --features 64 --batch-size 128
```

`python -m kernelyra packs path` показывает путь к редактируемой JSON-таблице.
Kernelyra валидирует её при каждом чтении. Ограничения аллокаций, границы
контекста и Model Guard являются обязательными защитами и из пака не удаляются.

## Python

```python
from kernelyra import Engine

with Engine("./project") as kernelyra:
    health = kernelyra.doctor("train.csv", "label")
    result = kernelyra.fit("train.csv", "label", backend="auto")
    report = kernelyra.report(result.run.id, "./project/run-report.json")

print(report["output"])
```

Чтобы использовать имена команд PowerShell из исходной копии проекта:

```powershell
Import-Module .\powershell\Kernelyra.psd1 -Force
Test-KernelyraDataset .\data\train.csv -Target label
Get-KernelyraDataContract .\data\train.csv -Target label
Get-KernelyraNativeStatus
Get-KernelyraCpuTuning -Records 100000 -Features 32 -BatchSize 64
Get-KernelyraPack | Format-Table Name, Base, Built_In, Algorithms
Copy-KernelyraPack my-careful -Base careful
Add-KernelyraPackAlgorithm my-careful thread_parallel_gradient
Get-KernelyraCpuTuning -Pack my-careful -Records 100000 -Features 64 -BatchSize 128
Get-KernelyraPlan .\data\train.csv -Target label
Start-KernelyraTraining .\data\train.csv -Target label -Execution cpu -Pack throughput -Cpu 100 -Ram 85 -Threads 12
```

## Воспроизводимый CPU-бенчмарк

Один локальный однопоточный тест на MSI Modern 14 C12M (Intel Core i5-1235U,
Windows 11, Python 3.12.10, NumPy 2.1.3) содержит девять чередующихся прогонов
после одного прогрева. В нём две узкие float32-нагрузки; это не общий рейтинг
фреймворков.

| Нагрузка | Медиана Kernelyra | Медиана NumPy | Результат только для этой машины |
|---|---:|---:|---|
| 1 000 одинаковых logistic updates, 8 192 × 64 | 0,1337745 с | 0,1300291 с | NumPy быстрее в `1,029×`; accuracy одинакова (99,0356%) |
| Copy → impute → normalize → clip, 8 192 × 64 | 0,0022132 с | 0,0023485 с | Kernelyra быстрее в `1,061×`; 75 repaired values и одинаковый checksum |

В случае обучения финальная accuracy у native и NumPy совпадает, абсолютная
разница финального loss — `1,49e-8`. Отдельная нетаймируемая проверка
random-batch пути увидела все пять native-ролей (`mask: 31`). Это доказательство
участия компонентов, но оно не входит в измерение скорости. Эти результаты не
доказывают скорость на другом железе, датасете, модели, лимите памяти или
нагрузке. См. [методику и runner](benchmarks/cpu/README.md) и
[сырой JSON девяти прогонов](benchmarks/cpu/results/modern-14-c12m-v0.6.0a1.json).

<a id="limits"></a>

## Ограничения и граница roadmap

Проверяемая цель релиза — **Windows x64 с Python 3.11–3.13**. Kernelyra 0.6
обучает только табличные модели. В ней нет встроенных тренеров для LLM,
изображений, аудио, видео, 3D и других модальностей. Распознавание расширения
не означает, что его можно извлечь, обучать на нём модель или использовать как
поддерживаемый model container.
`hybrid` требует обнаруженного ускорителя и установленного совместимого
опционального backend; это не обещание, что встроенный native-backend обучает
на любой GPU.

## Дополнительно

Выполни `python -m kernelyra --help` для списка команд и
`python -m kernelyra formats` для уровней возможностей. Также см. [changelog](CHANGELOG.md),
[правила участия](CONTRIBUTING.md), [security policy](SECURITY.md),
[уведомления о сторонних компонентах](THIRD_PARTY_NOTICES.md) и [лицензию](LICENSE).
