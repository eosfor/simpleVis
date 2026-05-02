# Pokemon Yard Simulation

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eosfor/simpleVis/HEAD?urlpath=lab/tree/pokemon_yard_simulation.ipynb)

Jupyter/Manim-модель школьного двора с карточками Pokemon: стартовая удача дает больше будущих возможностей обмена, даже если одна конкретная возможность устроена одинаково для всех.

## Запуск в Binder

Откройте кнопку **Open in Binder** выше. Binder соберет окружение и сразу откроет `pokemon_yard_simulation.ipynb`.

Binder использует:

- `requirements.txt` для Python-зависимостей: `numpy`, `pandas`, `matplotlib`, `jupyterlab`, `manim`.
- `apt.txt` для системных пакетов, нужных Manim и ffmpeg.

Прямая ссылка должна иметь такой вид:

```text
https://mybinder.org/v2/gh/eosfor/simpleVis/HEAD?urlpath=lab/tree/pokemon_yard_simulation.ipynb
```

## Что внутри

- `pokemon_yard_simulation.ipynb` - основной notebook с графиками стартового распределения, симуляцией обменов и метриками.
- `pokemon_yard_model.py` - воспроизводимая модель обменов.
- `manim_card_flow.py` - Manim-сцена для анимации перетекания карточек.
- `requirements.txt` и `apt.txt` - конфигурация Binder.

## Локальный запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
jupyter lab pokemon_yard_simulation.ipynb
```

## Рендер Manim

В Binder используйте команду без `-p`: флаг `-p` пытается открыть видео через системный viewer (`xdg-open`), которого в Binder обычно нет.

```bash
manim -ql manim_card_flow.py CardFlowScene
manim -ql manim_card_flow.py CircleTradeScene
```

Локально можно открыть preview сразу после рендера:

```bash
manim -pql manim_card_flow.py CardFlowScene
manim -pql manim_card_flow.py CircleTradeScene
```

Более высокое качество:

```bash
manim -pqh manim_card_flow.py CardFlowScene
manim -pqh manim_card_flow.py CircleTradeScene
```
