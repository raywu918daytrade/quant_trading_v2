"""LINE Flex 卡片積木 (Python 版), 和 line_bot/worker.js 的 card / kv / stockRow / tile 同一套外觀。
所有推播 (orb_live 的訊號 / 提醒 / 結算 / 不交易 / 出錯、fix_close 的改正通知、line-test) 都用這裡組卡片, 樣式改這一個檔就好。
card() 回傳一個 LINE message 物件 (type=flex), 直接丟給 orb_live.Line.push()。"""
import math

UP, DN, INK, INK2, SEP, TILE = "#d33a33", "#1a9e6a", "#121212", "#6b6a65", "#ecece8", "#f6f5f1"
ACCENT = {"entry": "#1f6fd1", "stop": "#e07a00", "info": "#8a8984", "ok": "#1a9e6a", "warn": "#e07a00", "error": "#d33a33", "sum": "#121212"}
SITE = "https://fade.just1stock.com/web/"


def ok(v):
    try: return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError): return False

def f2(v): return f"{float(v):.2f}".rstrip("0").rstrip(".") if ok(v) else "–"
def yuan(v): return f"{float(v):+,.0f}" if ok(v) else "–"
def pct(v): return f"{float(v):+.2f}%" if ok(v) else "–"
def color(v): return UP if ok(v) and v > 0 else DN if ok(v) and v < 0 else INK
def chart_url(d, sid): return f"{SITE}chart.html?d={d}&s={sid}"


def sep(): return {"type": "separator", "color": SEP, "margin": "md"}
def label(t): return {"type": "text", "text": t, "size": "xs", "weight": "bold", "color": INK2, "margin": "md"}
def note(t): return {"type": "text", "text": str(t)[:2000], "size": "sm", "color": INK2, "wrap": True, "margin": "sm"}
def small(t): return {"type": "text", "text": str(t)[:2000], "size": "xxs", "color": INK2, "wrap": True, "margin": "sm"}

def kv(k, v, c=INK):
    return {"type": "box", "layout": "horizontal", "margin": "sm", "contents": [
        {"type": "text", "text": str(k), "size": "sm", "color": INK2, "flex": 4},
        {"type": "text", "text": str(v if v not in (None, "") else "–"), "size": "sm", "color": c, "flex": 6, "align": "end", "wrap": True}]}

def tile(name, value, sub="", c=None):
    """大數字格子; value 是字串 (已格式化), c 是顏色。"""
    return {"type": "box", "layout": "vertical", "flex": 1, "backgroundColor": TILE, "cornerRadius": "8px", "paddingAll": "8px", "contents": [
        {"type": "text", "text": name, "size": "xxs", "color": INK2, "wrap": True},
        {"type": "text", "text": str(value), "size": "lg", "weight": "bold", "color": c or INK, "adjustMode": "shrink-to-fit"},
        *([{"type": "text", "text": sub, "size": "xxs", "color": INK2, "wrap": True}] if sub else [])]}

def tiles(*ts): return {"type": "box", "layout": "horizontal", "spacing": "md", "contents": list(ts)}

def row(title, sub, right, right_sub="", rc=INK, uri=None):
    """一檔一行: 左邊 名稱 + 小字說明, 右邊 數字 + 小字; uri 有值時整行可點 (開 K 線)。"""
    b = {"type": "box", "layout": "horizontal", "margin": "md", "contents": [
        {"type": "box", "layout": "vertical", "flex": 7, "contents": [
            {"type": "text", "text": title, "size": "sm", "weight": "bold", "color": INK, "wrap": True},
            {"type": "text", "text": sub, "size": "xxs", "color": INK2, "wrap": True}]},
        {"type": "box", "layout": "vertical", "flex": 3, "contents": [
            {"type": "text", "text": right, "size": "sm", "weight": "bold", "color": rc, "align": "end"},
            *([{"type": "text", "text": right_sub, "size": "xxs", "color": rc, "align": "end"}] if right_sub else [])]}]}
    if uri: b["action"] = {"type": "uri", "label": "K線", "uri": uri}
    return b

def card(title, blocks, sub="", footer=None, hint="", kind="info", alt=""):
    """title 標題; sub 標題下小字; blocks body 積木; footer = (標籤, 網址) 或其 list; hint 標題右側小字; kind 決定標題上方色條。
    alt 是通知列預覽文字 (上限 400 字), 不給就用 title + sub。"""
    fl = [footer] if isinstance(footer, tuple) else (footer or [])
    bubble = {"type": "bubble", "size": "mega",
              "header": {"type": "box", "layout": "vertical", "paddingBottom": "6px", "contents": [
                  {"type": "box", "layout": "vertical", "height": "3px", "backgroundColor": ACCENT.get(kind, ACCENT["info"]), "cornerRadius": "2px", "contents": []},
                  {"type": "box", "layout": "horizontal", "margin": "md", "contents": [
                      {"type": "text", "text": title, "weight": "bold", "size": "md", "color": INK, "flex": 1, "wrap": True},   # 長標題換行, 不截成「…」
                      *([{"type": "text", "text": hint, "size": "xxs", "color": INK2, "align": "end", "gravity": "center", "flex": 0}] if hint else [])]},
                  *([{"type": "text", "text": sub, "size": "xxs", "color": INK2, "wrap": True}] if sub else [])]},
              "body": {"type": "box", "layout": "vertical", "paddingTop": "0px", "contents": blocks or [note(" ")]}}
    if fl:
        bubble["footer"] = {"type": "box", "layout": "vertical", "paddingTop": "0px", "contents": [
            {"type": "button", "style": "link", "height": "sm", "action": {"type": "uri", "label": lb[:40], "uri": u}} for lb, u in fl[:4]]}
    return {"type": "flex", "altText": (alt or f"{title} {sub}").strip()[:400], "contents": bubble}

def alert(title, what, impact="", kind="warn", when=None, detail="", footer=None):
    """警示 / 出錯卡片: 短標題 + 時間 / 狀況, 影響用整行左對齊的說明 (+ 小字錯誤細節), 不要把整句塞進標題。when 預設現在 (台北)。"""
    if when is None:
        from datetime import datetime; from zoneinfo import ZoneInfo
        when = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%H:%M:%S")
    blocks = ([kv("時間", when)] if when else []) + [kv("狀況", what)] + ([note("影響: " + impact)] if impact else []) + ([small(detail)] if detail else [])
    alt = f"{title} {when}: {what}" + (f"。{impact}" if impact else "") + (f" [{detail}]" if detail else "")
    return card(title, blocks, kind=kind, footer=footer, alt=alt)

def from_text(text, kind="info"):
    """舊式純文字 → 卡片 (給 line-test --text 這類自由文字): 第一行當標題 (太長會換行), 其餘當內文。
    固定的警示請用 alert(), 不要整句丟進來。"""
    lines = str(text).strip().split("\n")
    return card(lines[0][:60], [note("\n".join(lines[1:]))] if len(lines) > 1 else [], kind=kind, alt=str(text)[:400])
