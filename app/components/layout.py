"""Page furniture: header, section headings, stat row, radio card, placeholder card, score table, bars."""
from html import escape

import streamlit as st

_RADIO_ICON = '<i></i><i></i><i></i><i></i><i></i>'   # sound bars, drawn in CSS (st.html drops inline SVG)


def page_header(title, kicker='', meta=''):
    st.html(f'<div class="bb-head"><div><div class="bb-kicker">{escape(kicker)}</div>'
            f'<h1 class="bb-title">{escape(title)}</h1></div><div class="bb-meta">{escape(meta)}</div></div>')


def section(title, note=''):
    st.html(f'<div class="bb-section"><h3>{escape(title)}</h3>{f"<span>{escape(note)}</span>" if note else ""}</div>')


def stats(items, vertical=False):
    """items: dicts with value, label, optional unit and key (True paints the value red; use once per page).
    vertical: stack them in a column (beside a card) instead of a row."""
    cells = ''
    for i in items:
        unit = f'<small>{escape(i["unit"])}</small>' if i.get('unit') else ''
        key = ' key' if i.get('key') else ''
        cells += (f'<div class="bb-stat"><div class="v{key}">{escape(str(i["value"]))}{unit}</div>'
                  f'<div class="l">{escape(i["label"])}</div></div>')
    st.html(f'<div class="bb-stats{" vert" if vertical else ""}">{cells}</div>')


def radio(message, who='Pit wall radio'):
    st.html(f'<div class="bb-radio"><span class="ic">{_RADIO_ICON}</span><div><div class="who">{escape(who)}</div>'
            f'<div class="msg">{escape(message)}</div></div></div>')


def call_card(headline, points, who='Pit wall radio'):
    """The radio card with a headline (the call) and a few short lines under it (why, what it is worth, how sure)."""
    items = ''.join(f'<li><b>{escape(k)}</b>{escape(v)}</li>' for k, v in points)
    st.html(f'<div class="bb-radio"><span class="ic">{_RADIO_ICON}</span><div><div class="who">{escape(who)}</div>'
            f'<div class="msg">{escape(headline)}</div><ul class="bb-why">{items}</ul></div></div>')


def call_panel(kicker, headline, chips_html, blocks, who='Pit wall radio'):
    """The radio card as a panel: a headline with the strategy as tyre chips, then labelled blocks of pills and
    meters (blocks: (label, inner html) pairs, html built by the caller from escaped text)."""
    grid = ''.join(f'<div class="blk"><div class="lab">{escape(label)}</div>{inner}</div>' for label, inner in blocks)
    st.html(f'<div class="bb-radio bb-call"><span class="ic">{_RADIO_ICON}</span><div class="body"><div class="who">{escape(who)}</div>'
            f'<div class="msg"><span class="kick">{escape(kicker)}</span>{escape(headline)}{chips_html}</div>'
            f'<div class="grid">{grid}</div></div></div>')


def tiles(items):
    """A row of result tiles: items are dicts with label, value, optional sub (small text under the value), tone
    ('good', 'bad' or '') and bar (0 to 1, a small meter under the value)."""
    cells = ''
    for i in items:
        bar = (f'<span class="m"><i style="width:{max(0.0, min(1.0, i["bar"])) * 100:.0f}%"></i></span>'
               if i.get('bar') is not None else '')
        sub = f'<span class="s">{escape(i["sub"])}</span>' if i.get('sub') else ''
        cells += (f'<div class="bb-tile {i.get("tone", "")}"><span class="l">{escape(i["label"])}</span>'
                  f'<span class="v">{escape(i["value"])}</span>{bar}{sub}</div>')
    st.html(f'<div class="bb-tiles">{cells}</div>')


def soon(entry):
    """Placeholder body for a page still being built (registry status 'soon')."""
    items = ''.join(f'<li>{escape(x)}</li>' for x in entry.plans)
    st.html(f'<div class="bb-soon"><h4>{escape(entry.blurb)}</h4>'
            f'{f"<ul>{items}</ul>" if items else ""}</div>')


def score_table(rows, columns, best=(), base=(), note=''):
    """HTML table. rows: list of dicts; columns: list of (field, header, format). Fields in `best` get the
    minimum painted purple; rows whose 'Model' is in `base` are dimmed. A row's 'Tag' shows after its name."""
    mins = {f: min(r[f] for r in rows if r.get(f) is not None) for f in best}
    head = ''.join(f'<th>{escape(h)}</th>' for _, h, _ in columns)
    body = ''
    for r in rows:
        cells = ''
        for i, (f, _, fmt) in enumerate(columns):
            v = r.get(f)
            text = '' if v is None else (fmt.format(v) if fmt else escape(str(v)))
            if i == 0 and r.get('Tag'):
                text += f'<span class="tag">{escape(r["Tag"])}</span>'
            cls = ' class="best"' if f in mins and v is not None and v == mins[f] else ''
            cells += f'<td{cls}>{text}</td>'
        body += f'<tr{" class=base" if r.get("Model") in base else ""}>{cells}</tr>'
    st.html(f'<div class="bb-table-wrap"><table class="bb-table"><thead><tr>{head}</tr></thead><tbody>{body}'
            f'</tbody></table></div>{f"<p class=bb-note>{escape(note)}</p>" if note else ""}')


def bars(items, fmt='{:+.2f} s', note=''):
    """Horizontal bars scaled to the largest value. items: list of (label, value)."""
    top = max(abs(v) for _, v in items) or 1
    rows = ''.join(f'<div class="bb-bar"><span class="l">{escape(l)}</span><span class="t"><i style="width:{abs(v) / top * 100:.1f}%">'
                   f'</i></span><span class="n">{fmt.format(v)}</span></div>' for l, v in items)
    st.html(f'<div class="bb-bars">{rows}</div>{f"<p class=bb-note>{escape(note)}</p>" if note else ""}')


def definitions(pairs):
    """Label and text pairs as a two column list (methods)."""
    st.html('<dl class="bb-dl">' + ''.join(f'<dt>{escape(k)}</dt><dd>{escape(v)}</dd>' for k, v in pairs) + '</dl>')
