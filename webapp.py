"""Telegram Mini App для админ-панели «Мудрого Мха».

Страница отдаётся тем же aiohttp-сервером, что и /health, поэтому отдельный
сервис не нужен. Доступ проверяется по подписи initData от Telegram: пускаем
только владельца (OWNER_ID), остальным — 403.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import parse_qsl


def verify_init_data(
    init_data: str, bot_token: str, owner_id: int, max_age: int = 86400
) -> bool:
    """Проверяет подпись Telegram WebApp initData и что это владелец."""
    if not init_data or not bot_token:
        return False
    try:
        pairs = dict(parse_qsl(init_data, strict_parsing=True))
    except ValueError:
        return False

    received_hash = pairs.pop("hash", "")
    if not received_hash:
        return False

    data_check = "\n".join(f"{key}={pairs[key]}" for key in sorted(pairs))
    secret = hmac.new(
        b"WebAppData", bot_token.encode(), hashlib.sha256
    ).digest()
    calculated = hmac.new(
        secret, data_check.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(calculated, received_hash):
        return False

    try:
        user = json.loads(pairs.get("user") or "{}")
    except ValueError:
        return False
    if int(user.get("id") or 0) != int(owner_id):
        return False

    try:
        auth_date = int(pairs.get("auth_date") or 0)
    except ValueError:
        return False
    if auth_date <= 0 or time.time() - auth_date > max_age:
        return False
    return True


ADMIN_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Мудрый Мох · Админка</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<style>
  *{box-sizing:border-box}
  :root{color-scheme:light dark}
  body{margin:0;padding:14px 14px 28px;
    font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background:var(--tg-theme-bg-color,#f4f4f5);color:var(--tg-theme-text-color,#111)}
  .head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}
  .brand{font-size:17px;font-weight:700}
  .small{font-size:12px}
  .muted{color:var(--tg-theme-hint-color,#6b7280)}
  .icon{width:38px;height:38px;border:0;border-radius:10px;font-size:18px;
    background:var(--tg-theme-secondary-bg-color,#e5e7eb);color:inherit}
  .tabs{display:flex;gap:6px;margin-bottom:12px;overflow-x:auto;padding-bottom:2px}
  .tabs button{flex:0 0 auto;border:0;border-radius:9px;padding:8px 12px;font:inherit;
    background:var(--tg-theme-secondary-bg-color,#e5e7eb);color:inherit;opacity:.75}
  .tabs button.active{background:var(--tg-theme-button-color,#2ea6ff);
    color:var(--tg-theme-button-text-color,#fff);opacity:1}
  section{display:none}
  section.active{display:block}
  .kpis{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
  .kpi{background:var(--tg-theme-secondary-bg-color,#ececef);border-radius:12px;padding:10px 12px}
  .kpi .v{font-size:21px;font-weight:700;line-height:1.1}
  .kpi .l{font-size:11px;margin-top:3px;color:var(--tg-theme-hint-color,#6b7280)}
  .card{background:var(--tg-theme-secondary-bg-color,#ececef);border-radius:12px;
    padding:12px;margin-bottom:10px}
  .card h3{margin:0 0 10px;font-size:13px;font-weight:650;
    color:var(--tg-theme-hint-color,#6b7280);text-transform:uppercase;letter-spacing:.03em}
  .row{margin-bottom:9px}
  .row:last-child{margin-bottom:0}
  .row .top{display:flex;justify-content:space-between;gap:8px;font-size:13px}
  .bar{height:7px;border-radius:5px;background:rgba(127,127,127,.18);overflow:hidden;margin-top:4px}
  .bar>i{display:block;height:100%;border-radius:5px;background:var(--tg-theme-button-color,#2ea6ff)}
  .list{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:8px 0;
    border-bottom:1px solid rgba(127,127,127,.14)}
  .list:last-child{border-bottom:0}
  .list>span{min-width:0}
  .list b{overflow-wrap:anywhere}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
    background:rgba(127,127,127,.16)}
  .ok{color:#16a34a}.bad{color:#dc2626}.warn{color:#d97706}
  .btns{display:flex;gap:8px;flex-wrap:wrap}
  .btn{border:0;border-radius:10px;padding:10px 14px;font:inherit;font-weight:600;
    background:var(--tg-theme-secondary-bg-color,#e5e7eb);color:inherit}
  .btn.primary{background:var(--tg-theme-button-color,#2ea6ff);
    color:var(--tg-theme-button-text-color,#fff)}
  .btn.ghost{opacity:.8}
  .btn.mini{padding:5px 9px;font-size:12px;border-radius:8px}
  .btn.danger{background:rgba(220,38,38,.16);color:#dc2626}
  .choice{display:flex;gap:6px;flex-wrap:wrap}
  .choice button{border:0;border-radius:9px;padding:7px 10px;font:inherit;font-size:13px;
    background:rgba(127,127,127,.16);color:inherit}
  .choice button.active{background:var(--tg-theme-button-color,#2ea6ff);
    color:var(--tg-theme-button-text-color,#fff)}
  .choice button:disabled{opacity:.45;text-decoration:line-through}
  .form{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
  .form input,.form select{border:0;border-radius:9px;padding:9px 10px;font:inherit;
    background:rgba(127,127,127,.16);color:inherit}
  .form input{flex:1 1 110px;min-width:0}
  .form select{flex:0 0 auto}
  .empty{font-size:12px;color:var(--tg-theme-hint-color,#6b7280)}
  .req{font-size:12px;padding:7px 0;border-bottom:1px solid rgba(127,127,127,.14)}
  .req:last-child{border-bottom:0}
  .req .meta{display:flex;justify-content:space-between;gap:8px}
  .req .preview{color:var(--tg-theme-hint-color,#6b7280);margin-top:2px;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);
    background:rgba(30,30,30,.92);color:#fff;padding:9px 14px;border-radius:10px;
    font-size:12px;opacity:0;pointer-events:none;transition:opacity .2s;max-width:90vw;
    text-align:center;z-index:9}
  #toast.show{opacity:1}
</style>
</head>
<body>
<div class="head">
  <div>
    <div class="brand">🌿 Мудрый Мох</div>
    <div class="small muted" id="build">админка</div>
  </div>
  <button class="icon" id="refresh" title="Обновить">⟳</button>
</div>

<nav class="tabs">
  <button data-tab="overview" class="active">Обзор</button>
  <button data-tab="models">Модели</button>
  <button data-tab="people">Люди</button>
  <button data-tab="requests">Запросы</button>
  <button data-tab="groups">Группы</button>
  <button data-tab="settings">Настройки</button>
</nav>

<section id="tab-overview" class="active">
  <div class="kpis" id="kpis"></div>
  <div class="card">
    <h3>Токены по моделям</h3>
    <div id="models-chart"></div>
  </div>
  <div class="card">
    <h3>Балансы</h3>
    <div id="balances"></div>
  </div>
  <div class="btns">
    <button class="btn primary" id="send-summary">📨 Отправить сводку</button>
    <button class="btn ghost" id="refresh2">🔄 Обновить</button>
  </div>
</section>

<section id="tab-models">
  <div class="card">
    <h3>Модели за 24 часа</h3>
    <div id="models-list"></div>
  </div>
  <div class="card">
    <h3>Доступно владельцу</h3>
    <div id="models-available"></div>
  </div>
</section>

<section id="tab-people">
  <div class="card">
    <h3>Выданные доступы</h3>
    <div id="grants-list"></div>
    <div class="form">
      <input id="grant-id" inputmode="numeric" placeholder="ID пользователя">
      <select id="grant-provider">
        <option value="deepseek">DeepSeek</option>
        <option value="qwen">Qwen</option>
      </select>
      <button class="btn primary mini" id="grant-add">Выдать</button>
    </div>
  </div>
  <div class="card">
    <h3>Топ пользователей</h3>
    <div id="top-users"></div>
  </div>
  <div class="card">
    <h3>Сервис</h3>
    <div id="service"></div>
  </div>
</section>

<section id="tab-requests">
  <div class="card">
    <h3>Последние запросы</h3>
    <div id="requests-list"></div>
    <div class="btns" style="margin-top:10px">
      <button class="btn ghost mini" id="req-more">Показать больше</button>
    </div>
  </div>
</section>

<section id="tab-groups">
  <div class="card">
    <h3>Белый список групп</h3>
    <div id="groups-list"></div>
    <div class="form">
      <input id="group-id" inputmode="numeric" placeholder="chat_id (-100…)">
      <input id="group-title" placeholder="Название">
      <button class="btn primary mini" id="group-add">Добавить</button>
    </div>
  </div>
  <div class="card">
    <h3>Заблокированные в ЛС</h3>
    <div id="blocked-list"></div>
  </div>
</section>

<section id="tab-settings">
  <div class="card">
    <h3>Модель</h3>
    <div class="choice" id="model-choice"></div>
  </div>
  <div class="card">
    <h3>Режимы</h3>
    <div class="row">
      <div class="top"><span>Rich-ответы</span><span class="pill" id="rich-state">—</span></div>
      <div class="btns" style="margin-top:8px">
        <button class="btn" id="toggle-rich">Переключить</button>
      </div>
    </div>
    <div class="row">
      <div class="top"><span>Deepthink</span><span class="pill" id="deepthink-state">—</span></div>
      <div class="choice" style="margin-top:8px" id="deepthink-choice"></div>
    </div>
    <div class="row">
      <div class="top"><span>Поиск</span><span class="pill" id="search-state">—</span></div>
      <div class="choice" style="margin-top:8px" id="search-choice"></div>
    </div>
  </div>
  <div class="card">
    <h3>Сборка</h3>
    <div class="small muted" id="build2">—</div>
  </div>
</section>

<div id="toast"></div>

<script>
const tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;
if (tg) { try { tg.ready(); tg.expand(); tg.setHeaderColor('secondary_bg_color'); } catch(e){} }
const initData = tg ? (tg.initData || '') : '';

const $ = (id) => document.getElementById(id);
const esc = (v) => String(v == null ? '' : v).replace(/[&<>"']/g, (c) =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function fmt(n) {
  n = Number(n || 0);
  if (!isFinite(n)) return '—';
  if (Math.abs(n) >= 1000) return n.toLocaleString('ru-RU');
  return Number.isInteger(n) ? String(n) : String(Math.round(n * 100) / 100);
}

function clock(iso) {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  return d.toLocaleTimeString('ru-RU', {hour: '2-digit', minute: '2-digit'});
}

let toastTimer = null;
function toast(text) {
  const el = $('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2600);
}

async function api(path, options) {
  const opts = options || {};
  const headers = Object.assign({
    'X-Init-Data': initData,
    'Content-Type': 'application/json'
  }, opts.headers || {});
  const res = await fetch(path, Object.assign({}, opts, {headers}));
  if (!res.ok) {
    let text = await res.text();
    try { text = JSON.parse(text).error || text; } catch(e) {}
    throw new Error(text || ('HTTP ' + res.status));
  }
  return res.json();
}

function bar(value, max) {
  const p = max > 0 ? Math.max(0, Math.min(100, value / max * 100)) : 0;
  return '<div class="bar"><i style="width:' + p.toFixed(1) + '%"></i></div>';
}

function confirmAsk(text) {
  return new Promise((resolve) => {
    if (tg && tg.showConfirm) tg.showConfirm(text, (ok) => resolve(!!ok));
    else resolve(window.confirm(text));
  });
}

let state = null;
let requestLimit = 30;

function render(data) {
  state = data;
  $('build').textContent = data.build || 'админка';
  $('build2').textContent = data.build || '—';

  const kpis = [
    ['Запросы', data.requests],
    ['Вызовы моделей', data.calls],
    ['Ошибки', data.errors],
    ['Кредиты', fmt(data.credits)],
  ];
  $('kpis').innerHTML = kpis.map(([l, v]) =>
    '<div class="kpi"><div class="v">' + esc(fmt(v)) + '</div><div class="l">' + esc(l) + '</div></div>'
  ).join('');

  const models = data.models || [];
  const peak = Math.max.apply(null, models.map(m => Number(m.tokens || 0)).concat([1]));
  $('models-chart').innerHTML = models.length ? models.map(m =>
    '<div class="row"><div class="top"><b>' + esc(m.model) + '</b>' +
    '<span class="muted">' + esc(fmt(m.tokens)) + ' ток.</span></div>' +
    bar(Number(m.tokens || 0), peak) + '</div>'
  ).join('') : '<div class="empty">Пока нет вызовов моделей.</div>';

  $('models-list').innerHTML = models.length ? models.map(m =>
    '<div class="list"><span><b>' + esc(m.model) + '</b>' +
    (m.provider ? ' <span class="pill">' + esc(m.provider) + '</span>' : '') +
    '</span><span class="muted">' + esc(fmt(m.calls)) + ' выз. · ' +
    esc(fmt(m.tokens)) + ' ток.</span></div>'
  ).join('') : '<div class="empty">Нет данных.</div>';

  const available = data.models_available || [];
  $('models-available').innerHTML = available.map(m =>
    '<div class="list"><span>' + esc(m.label) + '</span>' +
    (m.available ? '<span class="ok">доступна</span>'
                 : '<span class="bad">' + esc(m.reason || 'недоступна') + '</span>') +
    '</div>'
  ).join('') || '<div class="empty">Нет моделей.</div>';

  // ---- Люди: выданные доступы
  const grants = data.grants_list || [];
  $('grants-list').innerHTML = grants.length ? grants.map(g =>
    '<div class="list"><span><b>' + esc(g.label || g.user_id) + '</b>' +
    ' <span class="pill">' + esc(g.provider) + '</span>' +
    '<div class="small muted">сегодня: ' + esc(fmt(g.used)) +
    (g.limit ? ' / ' + esc(fmt(g.limit)) : '') + '</div></span>' +
    '<button class="btn danger mini" data-revoke="' + esc(g.user_id) + '" ' +
    'data-provider="' + esc(g.provider) + '">Убрать</button></div>'
  ).join('') : '<div class="empty">Доступы никому не выданы.</div>';

  $('top-users').innerHTML = (data.top_users || []).length ? data.top_users.map(u =>
    '<div class="list"><span>' + esc(u.label || u.user_id) + '</span>' +
    '<span class="muted">' + esc(fmt(u.requests)) + ' запр.</span></div>'
  ).join('') : '<div class="empty">Пока нет пользователей.</div>';

  $('service').innerHTML =
    '<div class="list"><span>Групп в белом списке</span><b>' + esc(fmt(data.groups)) + '</b></div>' +
    '<div class="list"><span>Заблокировано ЛС</span><b>' + esc(fmt(data.blocked)) + '</b></div>' +
    '<div class="list"><span>Доступы DeepSeek</span><b>' + esc(fmt(data.grants.deepseek)) + '</b></div>' +
    '<div class="list"><span>Доступы Qwen</span><b>' + esc(fmt(data.grants.qwen)) + '</b></div>' +
    '<div class="list"><span>Контекстов</span><b>' + esc(fmt(data.contexts)) + '</b></div>' +
    '<div class="list"><span>Средний ответ</span><b>' + esc(fmt(data.avg_latency)) + ' с</b></div>';

  $('balances').innerHTML = (data.balances || []).map(b => {
    if (b.error) return '<div class="list"><span>' + esc(b.name) + '</span><span class="bad">' + esc(b.error) + '</span></div>';
    const amount = b.amount == null ? 'нет данных' : fmt(b.amount);
    return '<div class="list"><span>' + esc(b.name) + '</span><b>' + esc(amount) + '</b></div>';
  }).join('') || '<div class="empty">Нет данных о балансах.</div>';

  // ---- Запросы
  $('requests-list').innerHTML = (data.requests_list || []).length
    ? data.requests_list.map(r => {
        const status = r.status === 'ok' ? '<span class="ok">ok</span>'
          : r.status === 'running' ? '<span class="warn">идёт</span>'
          : '<span class="bad">' + esc(r.status || 'err') + '</span>';
        const tokens = Number(r.tokens || 0);
        return '<div class="req"><div class="meta"><span><b>' + esc(r.label || r.user_id || '—') + '</b> ' +
          esc(r.model || '') + ' ' + status + '</span><span class="muted">' +
          esc(clock(r.created_at)) + (tokens ? ' · ' + esc(fmt(tokens)) + ' ток.' : '') +
          '</span></div><div class="preview">' + esc(r.preview || '') + '</div></div>';
      }).join('')
    : '<div class="empty">Пока нет запросов.</div>';

  // ---- Группы
  $('groups-list').innerHTML = (data.groups_list || []).length
    ? data.groups_list.map(g =>
        '<div class="list"><span><b>' + esc(g.title || g.chat_id) + '</b>' +
        '<div class="small muted"><code>' + esc(g.chat_id) + '</code></div></span>' +
        '<button class="btn danger mini" data-group-remove="' + esc(g.chat_id) + '">Убрать</button></div>'
      ).join('')
    : '<div class="empty">Список пуст.</div>';

  $('blocked-list').innerHTML = (data.blocked_list || []).length
    ? data.blocked_list.map(b =>
        '<div class="list"><span>' + esc(b.label || b.user_id) +
        (b.reason ? '<div class="small muted">' + esc(b.reason) + '</div>' : '') +
        '</span><button class="btn danger mini" data-unblock="' + esc(b.user_id) + '">Разблокировать</button></div>'
      ).join('')
    : '<div class="empty">Никто не заблокирован.</div>';

  // ---- Настройки
  const settings = data.settings || {};
  $('rich-state').textContent = settings.rich ? 'вкл' : 'выкл';
  $('deepthink-state').textContent = settings.deepthink || 'off';
  $('search-state').textContent = settings.search || 'auto';

  $('model-choice').innerHTML = available.map(m =>
    '<button data-model="' + esc(m.key) + '" class="' +
    (settings.model === m.key ? 'active' : '') + '"' +
    (m.available ? '' : ' disabled') + '>' + esc(m.label) + '</button>'
  ).join('');

  $('deepthink-choice').innerHTML = ['off','low','medium','high'].map(v =>
    '<button data-deepthink="' + v + '" class="' +
    (settings.deepthink === v ? 'active' : '') + '">' + v + '</button>'
  ).join('');

  $('search-choice').innerHTML = [
    ['auto','авто'], ['always','всегда'], ['off','выкл']
  ].map(([v, label]) =>
    '<button data-search="' + v + '" class="' +
    (settings.search === v ? 'active' : '') + '">' + label + '</button>'
  ).join('');
}

async function load() {
  try {
    const data = await api('/api/admin/overview?requests=' + requestLimit);
    render(data);
  } catch (e) {
    toast('Ошибка: ' + e.message);
  }
}

async function post(path, payload) {
  const data = await api(path, {method: 'POST', body: JSON.stringify(payload || {})});
  if (data.overview) render(data.overview);
  else load();
  if (data.message) toast(data.message);
  return data;
}

async function updateSetting(field, value) {
  try {
    await post('/api/admin/settings', {field: field, value: value});
  } catch (e) {
    toast('Ошибка: ' + e.message);
  }
}

document.querySelectorAll('.tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
  });
});

document.addEventListener('click', async (e) => {
  const model = e.target.closest('[data-model]');
  if (model) { updateSetting('model', model.dataset.model); return; }

  const deep = e.target.closest('[data-deepthink]');
  if (deep) { updateSetting('deepthink', deep.dataset.deepthink); return; }

  const search = e.target.closest('[data-search]');
  if (search) { updateSetting('search', search.dataset.search); return; }

  const revoke = e.target.closest('[data-revoke]');
  if (revoke) {
    const ok = await confirmAsk('Убрать доступ ' + revoke.dataset.provider + ' у ' + revoke.dataset.revoke + '?');
    if (!ok) return;
    try {
      await post('/api/admin/grant', {
        action: 'remove',
        user_id: Number(revoke.dataset.revoke),
        provider: revoke.dataset.provider
      });
    } catch (err) { toast('Ошибка: ' + err.message); }
    return;
  }

  const groupRemove = e.target.closest('[data-group-remove]');
  if (groupRemove) {
    const ok = await confirmAsk('Убрать группу ' + groupRemove.dataset.groupRemove + ' из белого списка?');
    if (!ok) return;
    try {
      await post('/api/admin/group', {
        action: 'remove',
        chat_id: Number(groupRemove.dataset.groupRemove)
      });
    } catch (err) { toast('Ошибка: ' + err.message); }
    return;
  }

  const unblock = e.target.closest('[data-unblock]');
  if (unblock) {
    try {
      await post('/api/admin/block', {action: 'remove', user_id: Number(unblock.dataset.unblock)});
    } catch (err) { toast('Ошибка: ' + err.message); }
    return;
  }
});

$('toggle-rich').addEventListener('click', () => updateSetting('rich', 'toggle'));
$('refresh').addEventListener('click', load);
$('refresh2').addEventListener('click', load);
$('req-more').addEventListener('click', () => { requestLimit += 30; load(); });

$('send-summary').addEventListener('click', async () => {
  try {
    const data = await api('/api/admin/summary', {method: 'POST', body: '{}'});
    toast(data.message || 'Сводка отправлена');
  } catch (e) {
    toast('Ошибка: ' + e.message);
  }
});

$('grant-add').addEventListener('click', async () => {
  const userID = Number(($('grant-id').value || '').trim());
  if (!userID || userID <= 0) { toast('Нужен числовой ID'); return; }
  try {
    await post('/api/admin/grant', {
      action: 'add',
      user_id: userID,
      provider: $('grant-provider').value
    });
    $('grant-id').value = '';
  } catch (e) { toast('Ошибка: ' + e.message); }
});

$('group-add').addEventListener('click', async () => {
  const chatID = Number(($('group-id').value || '').trim());
  if (!chatID) { toast('Нужен chat_id, например -1001234567890'); return; }
  try {
    await post('/api/admin/group', {
      action: 'add',
      chat_id: chatID,
      title: ($('group-title').value || '').trim()
    });
    $('group-id').value = '';
    $('group-title').value = '';
  } catch (e) { toast('Ошибка: ' + e.message); }
});

load();
setInterval(load, 30000);
</script>
</body>
</html>
"""
