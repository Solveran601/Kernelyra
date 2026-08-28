# Kernelyra: 10 пользовательских сценариев

Эта папка — не набор unit-тестов и не benchmark. Это десять коротких
сценариев, по которым разработчик или ранний пользователь может оценить
понятность Kernelyra 0.6.0a1: от проверки окружения до отчёта по завершённому
обучению.

Сценарии работают только с настоящими табличными наборами данных: CSV, TSV,
JSONL/NDJSON, NPZ или Parquet. Для обучения нужны бинарная или многоклассовая
классификация либо регрессия. Они **не** являются LLM-, image-, audio-, video-
или 3D-тренерами.

## Подготовка

Выполняй команды из корня исходного репозитория. Самый простой путь —
установить проект в его виртуальное окружение:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
```

Если Python 3.13 отсутствует, используй установленный Python 3.11 или 3.12.
Каждый PowerShell-сценарий сам берёт `./.venv/Scripts/python.exe`. Другой
интерпретатор можно передать так:

```powershell
.\examples\user-journeys\01-environment-and-capabilities.ps1 -Python C:\Python312\python.exe
```

Сценарий `09` использует local daemon. Если нужно, чтобы он запускал daemon
сам через `-AutoStart`, один раз добавь optional gateway extra:

```powershell
.\.venv\Scripts\python -m pip install -e ".[gateway]"
```

Все скрипты принимают `-Workspace`. По умолчанию это
`./.kernelyra-user-journeys`; там Kernelyra хранит локальные метаданные runs,
checkpoints и отчёты. Исходный датасет не копируется и не изменяется.

## Порядок прохождения

| № | Файл | Что пользователь проверяет | Пишет данные? |
|---:|---|---|---|
| 1 | `01-environment-and-capabilities.ps1` | установка, native status, реальные capabilities | только workspace |
| 2 | `02-data-doctor.ps1` | контракт данных и предупреждения до обучения | только workspace |
| 3 | `03-cpu-plan.ps1` | явные CPU/RAM/threads и итоговый план | только workspace |
| 4 | `04-backend-contracts.ps1` | какие backend/architecture действительно принимаются | только workspace |
| 5 | `05-native-cpu-training.ps1` | короткое CPU-обучение native backend | run и checkpoint |
| 6 | `06-custom-pack.ps1` | создание и изменение своего algorithm pack | только с `-Apply` |
| 7 | `07-python-fluent-api.py` | Python API `Config` + `Engine` | только с `--train` |
| 8 | `08-finetune-preflight.ps1` | проверка fine-tune входа и явный запуск | только с `-Apply` |
| 9 | `09-report-and-inference.ps1` | проверка завершённой модели и воспроизводимый отчёт | отчётный JSON |
| 10 | `10-jsonl-protocol.py` | нейтральный JSONL протокол для SDK | только workspace |

`04` не обучает модели: он намеренно показывает и успешные, и отклонённые
контракты. Это помогает не перепутать «распознано» с «реально реализовано».

## Быстрый пример

```powershell
$data = "C:\data\train.csv"
$target = "label"

.\examples\user-journeys\01-environment-and-capabilities.ps1
.\examples\user-journeys\02-data-doctor.ps1 -Dataset $data -Target $target
.\examples\user-journeys\03-cpu-plan.ps1 -Dataset $data -Target $target -Cpu 80 -Ram 70 -Threads 8
.\examples\user-journeys\05-native-cpu-training.ps1 -Dataset $data -Target $target -MaxSteps 100
```

После `05` скопируй ID run из вывода и проверь результат:

```powershell
.\examples\user-journeys\09-report-and-inference.ps1 -RunId <run-id> -Requests 20 -AutoStart
```

## Как собирать обратную связь

После каждого сценария запиши три вещи:

1. Понятно ли, какие данные нужны на вход и где появился результат?
2. Можно ли понять ошибку без чтения исходного кода?
3. Сколько параметров пришлось вспоминать или искать?

В первую очередь отмечай: неочевидный выбор backend/architecture, расхождение
между Data Doctor и планом, предупреждения о признаках, потребность в лишних
ручных шагах и неясные границы pause/resume. В 0.6.0a1 resume нельзя считать
точным восстановлением нейросетевого обучения: для критичных задач пока
сохраняй отдельный финальный checkpoint и запускай `09` после завершения run.

## Безопасность и границы

- `06` и `08` по умолчанию ничего не меняют; для изменения нужен `-Apply`.
- `05`, `08` и `09` создают локальные artefacts в указанном workspace.
- `09` обращается к локальному daemon. Передай `-AutoStart`, чтобы Kernelyra
  запустила его автоматически; без флага сценарий намеренно показывает эту
  зависимость через понятную ошибку.
- Встроенные паки не редактируются. `06` создаёт только пользовательский клон.
- `09` запускает inference только для завершённого run и проверяет, что
  checkpoint не изменился во время запроса.
- Скрипты не отправляют датасеты, модели или токены в сеть.
