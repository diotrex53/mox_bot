"""HTML-виджеты Telegram (Bot API 10.4, метод sendWidget).

Виджет — это одно HTML-сообщение до 32 КБ (и до 4 КБ после zlib), которое
Telegram рисует нативно. JS работает в песочнице без сети, поэтому данные
приходят только из HTML и chat_data.

Модуль ничего не знает про aiogram: на вход — обычные dict, на выходе —
(html, caption). Вся пользовательская текстовая информация экранируется,
чтобы модель не могла случайно (или намеренно) вставить разметку.
"""

from __future__ import annotations

import html as html_lib
import json
import re
import zlib

MAX_HTML_BYTES = 32768
MAX_COMPRESSED_BYTES = 4096
MAX_CAPTION = 1024

PALETTE = ("#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4")


class WidgetError(ValueError):
    """Виджет не проходит валидацию или слишком большой."""


def _esc(value) -> str:
    return html_lib.escape(str(value if value is not None else ""), quote=True)


def _clip(value, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _num(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _fmt(number) -> str:
    number = _num(number)
    if abs(number) >= 1000:
        return f"{number:,.0f}".replace(",", " ") if number.is_integer() else f"{number:,.1f}".replace(",", " ")
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _bar(value, maximum, width: int = 10) -> str:
    """Текстовая полоса для фолбэка, когда виджеты недоступны."""
    if maximum <= 0:
        return "░" * width
    ratio = max(0.0, min(1.0, _num(value) / _num(maximum)))
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def _js(value) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def _finalize(html_text: str, caption: str) -> tuple[str, str]:
    raw = html_text.encode("utf-8")
    if len(raw) > MAX_HTML_BYTES:
        raise WidgetError(
            f"виджет слишком большой ({len(raw)} байт из {MAX_HTML_BYTES}); "
            "сократи количество точек или строк"
        )
    packed = zlib.compress(raw, 9)
    if len(packed) > MAX_COMPRESSED_BYTES:
        raise WidgetError(
            f"виджет слишком большой после сжатия ({len(packed)} байт из "
            f"{MAX_COMPRESSED_BYTES}); сократи данные"
        )
    caption = _clip(caption, MAX_CAPTION)
    return html_text, caption


STYLE = """
<style>
*{box-sizing:border-box}
body{margin:0;font:14px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
color:var(--tg-theme-text-color,#111)}
.title{font-size:15px;font-weight:650;margin:0 0 8px}
.sub{color:var(--tg-theme-hint-color,#6b7280);font-size:12px;margin:0 0 10px}
.kpis{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
.kpi{background:var(--tg-theme-secondary-bg-color,#f3f4f6);border-radius:10px;padding:8px 10px}
.kpi .v{font-size:19px;font-weight:700;line-height:1.1}
.kpi .l{color:var(--tg-theme-hint-color,#6b7280);font-size:11px;margin-top:2px}
.bar{background:var(--tg-theme-secondary-bg-color,#e5e7eb);border-radius:5px;height:7px;
overflow:hidden;margin-top:4px}
.bar>i{display:block;height:100%;border-radius:5px;background:#3b82f6}
.row{margin:0 0 8px}
.row .top{display:flex;justify-content:space-between;gap:8px;font-size:12px}
.row .top b{font-weight:600}
.muted{color:var(--tg-theme-hint-color,#6b7280)}
button{display:block;width:100%;margin:6px 0 0;padding:9px 10px;border:0;border-radius:9px;
background:var(--tg-theme-secondary-bg-color,#f3f4f6);color:var(--tg-theme-text-color,#111);
font:inherit;text-align:left}
button.ok{background:#16a34a;color:#fff}
button.bad{background:#dc2626;color:#fff}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--tg-theme-secondary-bg-color,#e5e7eb);
vertical-align:top}
th{color:var(--tg-theme-hint-color,#6b7280);font-weight:600}
.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:11px;margin:2px 0 6px}
.legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px}
.msg{margin-top:8px;font-size:12px;min-height:16px}
button.task{display:flex;align-items:flex-start;gap:9px;padding:8px 10px}
button.task .box{flex:0 0 17px;width:17px;height:17px;border:1.5px solid #9ca3af;
border-radius:5px;margin-top:1px;text-align:center;line-height:15px;font-size:12px;color:transparent}
button.task .txt{flex:1 1 auto;min-width:0}
button.task.done .box{background:#16a34a;border-color:#16a34a;color:#fff}
button.task.done .txt{text-decoration:line-through;color:#6b7280}
</style>
"""


def _label_list(values, limit: int = 40, max_len: int = 12) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_clip(v, max_len) for v in values[:limit]]


def render_chart(spec: dict) -> tuple[str, str]:
    labels = _label_list(spec.get("labels"))
    raw_series = spec.get("series")
    if not labels and isinstance(raw_series, list) and raw_series:
        first = raw_series[0] if isinstance(raw_series[0], dict) else {}
        labels = _label_list(first.get("values"))
    series: list[dict] = []
    for index, item in enumerate(raw_series or []):
        if not isinstance(item, dict):
            continue
        values = [_num(v) for v in (item.get("values") or [])][:60]
        if not values:
            continue
        series.append(
            {
                "name": _clip(item.get("name") or f"Ряд {index + 1}", 24),
                "values": values,
                "color": PALETTE[index % len(PALETTE)],
            }
        )
    if not series:
        raise WidgetError("для графика нужен хотя бы один ряд values")

    count = min(len(series[0]["values"]), 60)
    if count < 1:
        raise WidgetError("в графике нет точек")
    series = [dict(s, values=s["values"][:count]) for s in series]
    if not labels or len(labels) < count:
        labels = [str(i + 1) for i in range(count)]
    labels = labels[:count]

    flat = [v for s in series for v in s["values"]]
    low, high = min(flat), max(flat)
    if high == low:
        high = low + 1 if low == 0 else abs(low) * 0.2
        low = low - (high - low) if low else 0
    pad = (high - low) * 0.08
    low -= pad
    high += pad

    width, height = 360, 232
    left, right, top, bottom = 40, 10, 22, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    chart_type = str(spec.get("chart_type") or "").lower()
    if chart_type not in {"line", "bar"}:
        chart_type = "bar" if (count <= 8 and len(series) == 1) else "line"

    def x_at(i: int) -> float:
        if chart_type == "bar":
            return left + plot_w * (i + 0.5) / count
        return left + (plot_w * i / max(1, count - 1))

    def y_at(v: float) -> float:
        return top + plot_h * (1 - (v - low) / (high - low))

    parts = [
        '<svg viewBox="0 0 %d %d" width="100%%" height="auto" role="img">' % (width, height),
        '<rect x="0" y="0" width="%d" height="%d" fill="none"/>' % (width, height),
    ]
    for step in range(5):
        v = low + (high - low) * step / 4
        y = y_at(v)
        parts.append(
            '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#9ca3af" '
            'stroke-opacity="0.25" stroke-width="1"/>' % (left, y, width - right, y)
        )
        parts.append(
            '<text x="%d" y="%.1f" font-size="9" fill="#6b7280" text-anchor="end">%s</text>'
            % (left - 5, y + 3, _esc(_fmt(round(v, 2))))
        )

    for s_index, item in enumerate(series):
        color = item["color"]
        if chart_type == "bar":
            bar_w = plot_w / count * (0.66 if len(series) == 1 else 0.74 / len(series))
            offset = (plot_w / count - bar_w * len(series)) / 2
            for i, value in enumerate(item["values"]):
                x = left + plot_w * i / count + offset + bar_w * s_index
                y = y_at(value)
                h = max(1.0, top + plot_h - y)
                parts.append(
                    '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2" fill="%s"/>'
                    % (x, y, bar_w, h, color)
                )
        else:
            points = " ".join(
                "%.1f,%.1f" % (x_at(i), y_at(v)) for i, v in enumerate(item["values"])
            )
            parts.append(
                '<polyline fill="none" stroke="%s" stroke-width="2" stroke-linejoin="round" '
                'stroke-linecap="round" points="%s"/>' % (color, points)
            )
            if count <= 14:
                for i, value in enumerate(item["values"]):
                    parts.append(
                        '<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s"/>'
                        % (x_at(i), y_at(value), color)
                    )

    label_step = max(1, count // 6)
    for i, label in enumerate(labels):
        if i % label_step and i not in (0, count - 1):
            continue
        parts.append(
            '<text x="%.1f" y="%d" font-size="9" fill="#6b7280" text-anchor="middle">%s</text>'
            % (x_at(i), height - 10, _esc(_clip(label, 10)))
        )
    parts.append("</svg>")

    unit = _clip(spec.get("unit") or "", 12)
    legend = ""
    if len(series) > 1:
        legend = '<div class="legend">' + "".join(
            f'<span><i style="background:{s["color"]}"></i>{_esc(s["name"])}</span>'
            for s in series
        ) + "</div>"
    title = _clip(spec.get("title") or "График", 60)
    html_text = (
        STYLE
        + f'<div class="title">{_esc(title)}</div>'
        + legend
        + "".join(parts)
        + (f'<div class="sub">{_esc(unit)}</div>' if unit else "")
    )
    return _finalize(html_text, title)


def render_table(spec: dict) -> tuple[str, str]:
    columns = _label_list(spec.get("columns"), limit=6, max_len=18)
    rows = spec.get("rows") or []
    if not isinstance(rows, list):
        rows = []
    clean_rows = []
    for row in rows[:10]:
        if isinstance(row, dict):
            row = list(row.values())
        if not isinstance(row, (list, tuple)):
            continue
        clean_rows.append([_clip(cell, 48) for cell in row][: len(columns) or 6])
    if not columns and clean_rows:
        columns = [str(i + 1) for i in range(len(clean_rows[0]))]
    if not columns:
        raise WidgetError("для таблицы нужны columns и rows")

    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in clean_rows
    )
    title = _clip(spec.get("title") or "Таблица", 60)
    html_text = (
        STYLE
        + f'<div class="title">{_esc(title)}</div>'
        + f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )
    return _finalize(html_text, title)


def render_quiz(spec: dict) -> tuple[str, str]:
    question = _clip(spec.get("question") or "Вопрос", 160)
    options = _label_list(spec.get("options"), limit=6, max_len=80)
    if len(options) < 2:
        raise WidgetError("для квиза нужно минимум 2 варианта")
    correct = spec.get("correct", 0)
    try:
        correct = int(correct)
    except (TypeError, ValueError):
        correct = 0
    if not 0 <= correct < len(options):
        correct = 0
    explanation = _clip(spec.get("explanation") or "", 220)
    title = _clip(spec.get("title") or "Проверь себя", 60)

    buttons = "".join(
        f'<button type="button" data-i="{i}">{_esc(text)}</button>'
        for i, text in enumerate(options)
    )
    html_text = (
        STYLE
        + f'<div class="title">{_esc(title)}</div>'
        + f'<div class="sub">{_esc(question)}</div>'
        + buttons
        + '<div class="msg" id="msg">&nbsp;</div>'
        + f"""<script>
(function(){{
  var correct={correct}, done=false, why={_js(explanation)};
  var msg=document.getElementById('msg');
  var btns=document.querySelectorAll('button[data-i]');
  function finish(i){{
    if(done) return; done=true;
    for(var k=0;k<btns.length;k++){{
      var n=parseInt(btns[k].getAttribute('data-i'),10);
      btns[k].className = n===correct ? 'ok' : (n===i ? 'bad' : '');
      btns[k].disabled=true;
    }}
    msg.className='msg';
    msg.textContent = (i===correct?'✅ Верно.':'❌ Неверно, правильный ответ: '+(correct+1)+'.')+(why?' '+why:'');
  }}
  for(var k=0;k<btns.length;k++){{
    btns[k].addEventListener('click',function(){{
      finish(parseInt(this.getAttribute('data-i'),10));
    }});
  }}
}})();
</script>"""
    )
    return _finalize(html_text, f"{title}\n{question}")


def render_progress(spec: dict) -> tuple[str, str]:
    items = spec.get("items") or []
    if not isinstance(items, list) or not items:
        raise WidgetError("для прогресса нужен список items")
    rows = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        label = _clip(item.get("label") or "Пункт", 40)
        value = _num(item.get("value"))
        maximum = _num(item.get("max"), 100) or 100
        percent = max(0.0, min(100.0, value / maximum * 100))
        rows.append(
            f'<div class="row"><div class="top"><b>{_esc(label)}</b>'
            f'<span class="muted">{_esc(_fmt(value))} / {_esc(_fmt(maximum))}</span></div>'
            f'<div class="bar"><i style="width:{percent:.1f}%"></i></div></div>'
        )
    if not rows:
        raise WidgetError("в прогрессе нет корректных пунктов")
    title = _clip(spec.get("title") or "Прогресс", 60)
    html_text = STYLE + f'<div class="title">{_esc(title)}</div>' + "".join(rows)
    return _finalize(html_text, title)


def render_checklist(
    title: str, tasks: list[str], note: str = ""
) -> tuple[str, str]:
    """Интерактивный чек-лист: галочки работают на клиенте до закрытия чата."""
    title = _clip(title or "Чек-лист", 60)
    clean = [_clip(str(task), 120) for task in tasks if str(task or "").strip()][:30]
    if not clean:
        raise WidgetError("для чек-листа нужен хотя бы один пункт")

    count = len(clean)
    rows = "".join(
        f'<button type="button" class="task" data-i="{i}">'
        f'<span class="box">✓</span><span class="txt">{_esc(text)}</span></button>'
        for i, text in enumerate(clean)
    )
    note_text = _clip(note, 90) or "Отметки живут, пока открыт чат."
    html_text = (
        STYLE
        + f'<div class="title">{_esc(title)}</div>'
        + f'<div class="sub" id="cnt">0 из {count}</div>'
        + rows
        + f'<div class="sub" style="margin-top:8px">{_esc(note_text)}</div>'
        + f"""<script>
(function(){{
  var total={count}, done=0;
  var cnt=document.getElementById('cnt');
  var btns=document.querySelectorAll('button.task');
  for(var k=0;k<btns.length;k++){{
    btns[k].addEventListener('click',function(){{
      this.classList.toggle('done');
      done=document.querySelectorAll('button.task.done').length;
      cnt.textContent=done+' из '+total+(done===total?' — готово 🎉':'');
    }});
  }}
}})();
</script>"""
    )
    return _finalize(
        html_text,
        f"{title} · {count} {_plural(count, 'пункт', 'пункта', 'пунктов')}",
    )


RENDERERS = {
    "chart": render_chart,
    "table": render_table,
    "quiz": render_quiz,
    "progress": render_progress,
}


def render_widget(kind: str, spec: dict) -> tuple[str, str]:
    renderer = RENDERERS.get((kind or "").strip().lower())
    if renderer is None:
        raise WidgetError(
            "неизвестный тип виджета; доступны: chart, table, quiz, progress"
        )
    if not isinstance(spec, dict):
        raise WidgetError("spec должен быть объектом")
    return renderer(spec)


def text_version(kind: str, spec: dict) -> str:
    """Простой текстовый вариант для групп и фолбэка."""
    spec = spec if isinstance(spec, dict) else {}
    title = _clip(spec.get("title") or "", 80)
    lines: list[str] = [f"<b>{_esc(title)}</b>"] if title else []
    kind = (kind or "").lower()
    if kind == "chart":
        labels = _label_list(spec.get("labels"), limit=20, max_len=16)
        series: list[dict] = []
        for item in (spec.get("series") or [])[:3]:
            if not isinstance(item, dict):
                continue
            values = [_num(v) for v in (item.get("values") or [])][:20]
            if values:
                series.append(
                    {
                        "name": _clip(item.get("name") or "Ряд", 24),
                        "values": values,
                    }
                )
        if len(series) == 1:
            values = series[0]["values"]
            peak = max((abs(v) for v in values), default=0) or 1
            for i, value in enumerate(values):
                label = labels[i] if i < len(labels) else str(i + 1)
                lines.append(
                    f"{_esc(_clip(label, 12))} {_bar(abs(value), peak)} {_esc(_fmt(value))}"
                )
        else:
            count = max((len(item["values"]) for item in series), default=0)
            for i in range(count):
                label = labels[i] if i < len(labels) else str(i + 1)
                parts = [
                    f"{_esc(item['name'])} {_esc(_fmt(item['values'][i]))}"
                    for item in series
                    if i < len(item["values"])
                ]
                lines.append(f"{_esc(_clip(label, 12))}: " + ", ".join(parts))
    elif kind == "table":
        columns = _label_list(spec.get("columns"), limit=5, max_len=24)
        raw_rows = spec.get("rows") or []
        clean_rows: list[list[str]] = []
        for row in raw_rows[:10]:
            if isinstance(row, dict):
                row = list(row.values())
            if isinstance(row, (list, tuple)):
                clean_rows.append([_clip(cell, 24) for cell in row][: len(columns) or 5])
        if clean_rows and not columns:
            columns = [str(i + 1) for i in range(len(clean_rows[0]))]
        if columns:
            widths = [len(str(c)) for c in columns]
            for row in clean_rows:
                for i, cell in enumerate(row[: len(widths)]):
                    widths[i] = max(widths[i], len(str(cell)))

            def fmt_row(cells: list[str]) -> str:
                padded = [str(c).ljust(widths[i]) for i, c in enumerate(cells[: len(widths)])]
                return " | ".join(padded).rstrip()

            table_lines = [fmt_row(columns)] + [fmt_row(row) for row in clean_rows]
            lines.append("<pre>" + _esc("\n".join(table_lines)) + "</pre>")
    elif kind == "quiz":
        lines.append(_esc(_clip(spec.get("question") or "", 200)))
        for i, option in enumerate(_label_list(spec.get("options"), limit=6, max_len=80)):
            lines.append(f"{i + 1}. {_esc(option)}")
        try:
            correct = int(spec.get("correct", 0)) + 1
        except (TypeError, ValueError):
            correct = 1
        lines.append(f"<i>Ответ: {correct}</i>")
    elif kind == "progress":
        for item in (spec.get("items") or [])[:8]:
            if not isinstance(item, dict):
                continue
            label = _clip(item.get("label") or "Пункт", 40)
            value = _num(item.get("value"))
            maximum = _num(item.get("max"), 100) or 100
            lines.append(
                f"• {_esc(label)}: {_bar(value, maximum)} "
                f"{_esc(_fmt(value))} / {_esc(_fmt(maximum))}"
            )
    elif kind == "checklist":
        for task in (spec.get("tasks") or [])[:30]:
            text = str(task or "").strip()
            if text:
                lines.append(f"☐ {_esc(_clip(text, 120))}")
    return "\n".join(line for line in lines if line).strip() or "Виджет недоступен."


def render_dashboard(data: dict, period_label: str = "24 часа") -> tuple[str, str]:
    """Дашборд админки: KPI + топ моделей + топ пользователей."""
    models = data.get("models") or []
    users = data.get("top_users") or []
    requests = int(data.get("requests") or 0)
    calls = int(data.get("calls") or 0)
    errors = int(data.get("request_errors") or 0) + int(data.get("call_errors") or 0)
    credits = _num(data.get("credits"))

    kpis = [
        ("Запросы", _fmt(requests)),
        ("Вызовы моделей", _fmt(calls)),
        ("Ошибки", _fmt(errors)),
        ("Кредиты", _fmt(credits)),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{_esc(v)}</div><div class="l">{_esc(l)}</div></div>'
        for l, v in kpis
    )

    max_tokens = max(
        (int(m.get("total_tokens") or 0) for m in models[:6]), default=0
    ) or 1
    model_rows = []
    for m in models[:6]:
        name = _clip(m.get("model") or "?", 28)
        tokens = int(m.get("total_tokens") or 0)
        percent = tokens / max_tokens * 100
        model_rows.append(
            f'<div class="row"><div class="top"><b>{_esc(name)}</b>'
            f'<span class="muted">{_esc(_fmt(tokens))} ток.</span></div>'
            f'<div class="bar"><i style="width:{percent:.1f}%"></i></div></div>'
        )
    if not model_rows:
        model_rows.append('<div class="sub">Пока нет вызовов моделей.</div>')

    user_rows = "".join(
        f'<div class="row"><div class="top"><b>{_esc(u.get("user_id") or "?")}</b>'
        f'<span class="muted">{_esc(_fmt(_num(u.get("requests"))))} запр.</span></div></div>'
        for u in users[:5]
    ) or '<div class="sub">Пока нет пользователей.</div>'

    html_text = (
        STYLE
        + f'<div class="title">🖥 Дашборд · {_esc(period_label)}</div>'
        + f'<div class="kpis">{kpi_html}</div>'
        + '<div class="sub">Токены по моделям</div>'
        + "".join(model_rows)
        + '<div class="sub" style="margin-top:10px">Топ пользователей</div>'
        + user_rows
    )
    return _finalize(html_text, f"Дашборд · {period_label}")


def dashboard_text(data: dict, period_label: str = "24 часа") -> str:
    """Текстовый дашборд для случая, когда Telegram запрещает виджеты."""
    models = data.get("models") or []
    users = data.get("top_users") or []
    requests = int(data.get("requests") or 0)
    calls = int(data.get("calls") or 0)
    errors = int(data.get("request_errors") or 0) + int(data.get("call_errors") or 0)
    credits = _num(data.get("credits"))

    lines = [
        f"🖥 <b>Дашборд · {_esc(period_label)}</b>",
        "",
        f"Запросы: <b>{_esc(_fmt(requests))}</b>",
        f"Вызовы моделей: <b>{_esc(_fmt(calls))}</b>",
        f"Ошибки: <b>{_esc(_fmt(errors))}</b>",
        f"Кредиты: <b>{_esc(_fmt(credits))}</b>",
        "",
        "<b>Токены по моделям</b>",
    ]
    peak = max((int(m.get("total_tokens") or 0) for m in models[:6]), default=0) or 1
    for item in models[:6]:
        name = _clip(item.get("model") or "?", 28)
        tokens = int(item.get("total_tokens") or 0)
        lines.append(
            f"{_esc(name)} {_bar(tokens, peak)} {_esc(_fmt(tokens))}"
        )
    if not models:
        lines.append("Пока нет вызовов моделей.")

    lines.extend(["", "<b>Топ пользователей</b>"])
    for user in users[:5]:
        lines.append(
            f"• <code>{_esc(user.get('user_id') or '?')}</code> — "
            f"{_esc(_fmt(_num(user.get('requests'))))}"
        )
    if not users:
        lines.append("Пока нет пользователей.")
    return "\n".join(lines)
