# Pokemon Yard Simulation

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eosfor/simpleVis/HEAD?urlpath=lab/tree/pokemon_yard_simulation.ipynb)

Jupyter/Manim-модель школьного двора с карточками Pokemon: стартовая удача дает больше будущих возможностей обмена, даже если одна конкретная сделка добровольна и субъективно выгодна обеим сторонам.

## Формулировка модели

У детей есть случайные стартовые коллекции карточек разной редкости, типа и рыночной ценности. У каждого ребенка свои субъективные предпочтения: кому-то нужны огненные карты, кому-то редкие, кому-то конкретный тип для воображаемой колоды.

Чем больше и разнообразнее коллекция, тем больше независимых возможностей обмена она создает: больше карточек можно предложить, больше детей могут захотеть что-то из твоей коллекции, и проще собрать bundle из нескольких менее нужных карт ради одной более нужной.

Сделка происходит только если обе стороны субъективно выигрывают. При этом владелец ликвидной карты не обязан принимать рыночный минус: если к нему приходят за нужной картой, он может запросить bundle с небольшим рыночным premium. Покупатель может согласиться, потому что субъективно эта карта для него ценнее, чем отданный набор.

Даже при таких добровольных обменах рыночная ценность коллекций может концентрироваться у тех, кто случайно получил лучшую стартовую позицию: у них больше leads, больше ликвидности и больше способов подобрать сделку.

`Gini` - индекс неравенства распределения: `0` означает полное равенство, `1` означает предельную концентрацию.

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

- `pokemon_yard_simulation.ipynb` - основной notebook с графиками стартового распределения, добровольными обменами и метриками.
- `pokemon_yard_model.py` - воспроизводимая модель карточек, предпочтений и добровольных bundle-сделок.
- `manim_card_flow.py` - Manim-сцены для анимации перетекания рыночной ценности.
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

В notebook Manim запускается тихо через `subprocess.run([sys.executable, "-m", "manim", ...])`, поэтому служебный `INFO`-вывод не показывается, а готовое видео встраивается прямо под ячейкой.

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
