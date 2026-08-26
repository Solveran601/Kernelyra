<p align="center">
  <img src="assets/brand/kernelyra-logo.png" alt="Kernelyra" width="520">
</p>

<p align="center">
  <a href="README.md">English version</a>
</p>

<p align="center"><img src="assets/brand/kernelyra-mark-animated.svg" alt="Анимированный знак Kernelyra" width="72"></p>

<p align="center"><strong>Нативное обучение табличных данных с контролем ресурсов.</strong></p>

Kernelyra **0.5.0a1 (V3 alpha)** — локальная библиотека для обучения табличных
моделей из терминала. Один и тот же путь планирования доступен в CLI, Python
API, PowerShell-модуле и JSONL-протоколе для поставляемых SDK.

<p align="center">
  <a href="#capabilities">Возможности</a> ·
  <a href="reports/CPU_BENCHMARK_2026-08-26.md">CPU-бенчмарк</a> ·
  <a href="#install">Установка</a> ·
  <a href="#powershell">PowerShell</a> ·
  <a href="#limits">Ограничения</a>
</p>

## CPU benchmark snapshot

<p align="center">
  <a href="reports/CPU_BENCHMARK_2026-08-26.md">Методика и результаты</a> ·
  <a href="reports/cpu-framework-matrix-2026-08-26.json">Исходный JSON</a> ·
  <a href="reports/CPU_BENCHMARK_2026-08-26.md#reproduce">Повторить запуск</a>
</p>

| Одинаковая float32 full-batch logistic-regression задача, только CPU | Результат |
| --- | --- |
| Машина и метод | Intel Core i5-1235U; один общий CPU-поток; 8 192 train + 2 048 hold-out строк; медиана 3 запусков |
| Качество | Kernelyra, NumPy, PyTorch, TensorFlow, JAX и Flax/Optax: **96.09% hold-out accuracy** |
| Kernelyra native | **14.67 мс** на 30 шагов — в 1.82× быстрее PyTorch в этой задаче |
| Текущая цель оптимизации | NumPy здесь **в 4.22× быстрее** Kernelyra; это не скрывается и не выдаётся за победу |

Матрица измеряет Kernelyra плюс NumPy, PyTorch, TensorFlow, JAX, Flax/Optax,
scikit-learn, River, XGBoost, LightGBM и CatBoost. Деревья и online-обучение
используют другие алгоритмы, поэтому они приведены без ложного общего рейтинга
скорости. Эти CPU-значения не являются заявлениями о GPU, LLM, изображениях или
универсальной производительности.

<a id="capabilities"></a>

## Что работает в этой alpha-версии

- Бинарная и многоклассовая классификация, а также регрессия на табличных
  данных.
- Прямое обучение на CSV, TSV, JSONL/NDJSON, числовых NPZ и опциональном
  Parquet.
- Встроенные native и NumPy backend; PyTorch и TensorFlow/Keras — опционально,
  когда они установлены.
- Явный выбор выполнения `cpu` или `hybrid`, заданные разработчиком лимиты
  CPU/RAM/GPU/потоков и четыре необязательных пакета алгоритмов (`careful`,
  `balanced`, `throughput`, `maximum`). Пакеты меняют ограниченные значения
  чанков/предзагрузки/рабочей памяти, но не классифицируют компьютер
  пользователя. Есть checkpoints, resume, отложенная проверка и восстановление
  лучшего checkpoint.
- Data Doctor: ограниченная предварительная проверка, подписанный контракт
  датасета, детерминированная рекомендация split и план неравномерных чанков.
- Model Guard V2: проверка конечности метрик и сохранение тренда качества в
  health-записи run.
- Переносимые JSON/HTML-отчёты эксперимента и явные лимиты CPU/RAM/GPU.

Data Doctor намеренно ограничен выборкой: выводы относятся к проверенным
строкам, а не ко всему датасету. Для materialized training классификация
разделяется детерминированно по классам, а time-like колонка сохраняет исходный
порядок входных строк. Обнаруженная group/context колонка пока является только
**предупреждением**: group-exclusive split в 0.5 ещё не реализован.

<a id="install"></a>

## Установка из исходников

Публикация на PyPI не настроена. После успешной сборки GitHub Actions для тега
Windows-артефакты этой alpha-версии прикрепляются к её GitHub pre-release.

```powershell
git clone https://github.com/Solveran601/Kernelyra.git
Set-Location Kernelyra
python -m pip install -e .
```

`.[data]` нужен для Parquet, `.[torch]` — для PyTorch, а `.[tensorflow]` — для
TensorFlow/Keras.

<a id="powershell"></a>

## Три команды PowerShell

```powershell
python -m kernelyra doctor
python -m kernelyra execution
python -m kernelyra plan .\data\train.csv --target label --execution cpu --pack throughput --cpu 100 --ram 85 --threads 12
```

При необходимости сначала выполни явную проверку данных:

```powershell
python -m kernelyra dataset doctor .\data\train.csv --target label
```

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
Get-KernelyraPlan .\data\train.csv -Target label
Start-KernelyraTraining .\data\train.csv -Target label -Execution cpu -Pack throughput -Cpu 100 -Ram 85 -Threads 12
```

<a id="limits"></a>

## Ограничения и граница roadmap

Проверяемая цель релиза — **Windows x64 с Python 3.11–3.13**. Kernelyra 0.5
обучает только табличные модели. В ней нет встроенных тренеров для LLM,
изображений, аудио, видео, 3D и других модальностей. Распознавание расширения
не означает, что его можно извлечь, обучать на нём модель или использовать как
поддерживаемый model container.
`hybrid` требует обнаруженного ускорителя и установленного совместимого
опционального backend; это не обещание, что встроенный native-backend обучает
на любой GPU.

V3 workflow benchmark измеряет локальную предварительную проверку и
планирование; это не сравнение с конкурентами и не универсальное заявление о
скорости. Точные входные данные и окружение сохранены в
[V3 JSON-отчёте](reports/v3-workflow-benchmark-2026-08-25.json).
Отдельный CPU-only matched-linear запуск сохраняет фактические результаты
Kernelyra, NumPy, PyTorch и JAX без ранжирования разных алгоритмов в
[framework matrix](reports/v3-framework-cpu-2026-08-25.json).
Более новая [CPU-матрица](reports/CPU_BENCHMARK_2026-08-26.md) фиксирует
Kernelyra и десять ML-библиотек на независимом hold-out наборе, оставляя
деревья и online-обучение вне линейного сравнения скорости.

## Дополнительно

Выполни `python -m kernelyra --help` для списка команд и
`python -m kernelyra formats` для уровней возможностей. Также см. [changelog](CHANGELOG.md),
[правила участия](CONTRIBUTING.md), [security policy](SECURITY.md),
[уведомления о сторонних компонентах](THIRD_PARTY_NOTICES.md) и [лицензию](LICENSE).
