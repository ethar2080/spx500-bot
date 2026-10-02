"""
بوت المضاربة اللحظية - SPX (عبر SPY) - فريم 3د - نسخة 【entity-GitHub¦canonical_name=GitHub】 Actions
نفس كودك الأصلي 100% بس بدون while True + حفظ الكولداون في ملف
"""
import os, json, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
TOKEN = os.getenv("TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID") or os.environ.get("CHAT_ID")
TD_KEY = os.getenv("TWELVE_KEY") or os.environ.get("TWELVE_KEY")

if not TOKEN or not CHAT_ID or not TD_KEY:
    raise SystemExit("Missing env: TELEGRAM_TOKEN, CHAT_ID, TWELVE_KEY")

SYMBOL = "SPY"
COOLDOWN_MIN = 30
MIN_AGREE = 1
DISCLAIMER = "⚠ اخلاء مسؤولية: يجب مراجعة الشارت، أنت من تقرر التداول"
STATE_FILE = "/tmp/bot_state.json"

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
        print(f"Telegram {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"Telegram error: {e}")

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1+up/dn)
def dmi(h,l,c,n=14):
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up>dn)&(up>0),up,0.0),index=c.index).ewm(alpha=1/n,adjust=False).mean()
    mdm = pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=c.index).ewm(alpha=1/n,adjust=False).mean()
    pdi, mdi = 100*pdm/atr, 100*mdm/atr
    dx = 100*(pdi-mdi).abs()/(pdi+mdi)
    return pdi, mdi, dx.ewm(alpha=1/n,adjust=False).mean()
def psar(h,l,start=0.01,inc=0.004,mx=1.0):
    n=len(h); out=np.zeros(n); out[0]=l[0]; bull,af,ep=True,start,h[0]
    for i in range(1,n):
        out[i]=out[i-1]+af*(ep-out[i-1])
        if bull:
            if l[i]<out[i]: bull, out[i], af, ep = False, ep, start, l[i]
            elif h[i]>ep: ep, af = h[i], min(af+inc,mx)
        else:
            if h[i]>out[i]: bull, out[i], af, ep = True, ep, start, h[i]
            elif l[i]<ep: ep, af = l[i], min(af+inc,mx)
    return out
def qqe_direction(c,n=14,factor=4.238,sm=5):
    rm_s = ema(rsi(c,n),sm); w=n*2-1
    dar = ema(ema((rm_s.shift(1)-rm_s).abs(),w),w)*factor
    rm = rm_s.values; dar = dar.fillna(0).values; N=len(c)
    lb,sb=np.zeros(N),np.zeros(N); tr=np.ones(N,dtype=int); out=np.zeros(N,dtype=int); d=0
    for i in range(1,N):
        nl,ns=rm[i]-dar[i], rm[i]+dar[i]
        lb[i]=max(lb[i-1],nl) if rm[i-1]>lb[i-1] and rm[i]>lb[i-1] else nl
        sb[i]=min(sb[i-1],ns) if rm[i-1]<sb[i-1] and rm[i]<sb[i-1] else ns
        if rm[i]>sb[i-1]: tr[i]=1
        elif rm[i]<lb[i-1]: tr[i]=-1
        else: tr[i]=tr[i-1]
        if tr[i]==1 and tr[i-1]==-1: d=1
        elif tr[i]==-1 and tr[i-1]==1: d=-1
        out[i]=d
    return pd.Series(out,index=c.index)

def get_1m():
    r=requests.get("https://api.twelvedata.com/time_series",
        params=dict(symbol=SYMBOL,interval="1min",outputsize=1500,timezone="America/New_York",order="ASC",apikey=TD_KEY),timeout=20).json()
    if "values" not in r:
        print(f"No values: {str(r)[:500]}"); return None
    df=pd.DataFrame(r["values"]); df["datetime"]=pd.to_datetime(df["datetime"])
    df=df.set_index("datetime").sort_index().astype(float)
    df.columns=[x.capitalize() for x in df.columns]; return df
def to_3m(df1):
    return df1.resample("3min").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

def analyse(df):
    c,h,l,v=df["Close"],df["High"],df["Low"],df["Volume"]
    pdi,mdi,adx=dmi(h,l,c); macd=ema(c,12)-ema(c,26); sig=macd.rolling(9).mean(); hist=macd-sig
    r14,e20=rsi(c,14),ema(c,20); sar=pd.Series(psar(h.values,l.values),index=c.index); q=qqe_direction(c)
    tp=(h+l+c)/3; day=df.index.date; vwap=(tp*v).groupby(day).cumsum()/v.groupby(day).cumsum()
    spike=(v>=1.5*v.rolling(20).mean()).astype(float); vol_ok=spike.rolling(3).max()>=1
    x=-1
    macd_bull=macd.iloc[x]>sig.iloc[x] and hist.iloc[x]>hist.iloc[x-1]
    macd_bear=macd.iloc[x]<sig.iloc[x] and hist.iloc[x]<hist.iloc[x-1]
    adx_strong=adx.iloc[x]>30 and adx.iloc[x]>adx.iloc[x-1]
    adx_bull=adx_strong and pdi.iloc[x]>mdi.iloc[x]; adx_bear=adx_strong and mdi.iloc[x]>pdi.iloc[x]
    qqe_bull,q_bear=q.iloc[x]==1,q.iloc[x]==-1
    ps_bull,ps_bear=c.iloc[x]>sar.iloc[x], c.iloc[x]<sar.iloc[x]
    rs_bull,rs_bear=r14.iloc[x]>55, r14.iloc[x]<45
    bull_n=sum([macd_bull,adx_bull,ps_bull,rs_bull,qqe_bull]); bear_n=sum([macd_bear,adx_bear,ps_bear,rs_bear,q_bear])
    blue=c.iloc[x]>vwap.iloc[x] and c.iloc[x]>e20.iloc[x] and r14.iloc[x]>60 and bool(vol_ok.iloc[x])
    gray=c.iloc[x]<vwap.iloc[x] and c.iloc[x]<e20.iloc[x] and r14.iloc[x]<40 and bool(vol_ok.iloc[x])
    yellow, purple = bull_n>=4, bear_n>=4
    call_ag=[n for n,ok in (("MACD",macd_bull),("ADX",adx_bull),("QQE",qqe_bull)) if ok]
    put_ag=[n for n,ok in (("MACD",macd_bear),("ADX",adx_bear),("QQE",q_bear)) if ok]
    call_cd=[n for n,ok in (("زرقاء",blue),("صفراء",yellow)) if ok]
    put_cd=[n for n,ok in (("رمادية",gray),("بنفسجية",purple)) if ok]
    print(f"{df.index[x]} close={c.iloc[x]:.2f} bull={bull_n} bear={bear_n} call={call_ag}/{call_cd} put={put_ag}/{put_cd}")
    return {"CALL":(call_ag,call_cd),"PUT":(put_ag,put_cd)}

def build_msg(strategy,side,agree,candles):
    icon="🟢" if side=="CALL" else "🔴"
    lines=[f"الاستراتيجية: {strategy}",f"نوع الصفقة: {side} {icon}","فريم التداول: 3د",f"توافق المؤشرات: {' / '.join(agree)}"]
    if strategy=="ORB": lines.append(f"قوة الشموع: {' + '.join(candles)}")
    lines+=["",DISCLAIMER]; return "\n".join(lines)

def load_state():
    try:
        with open(STATE_FILE,"r") as f:
            d=json.load(f); return {tuple(k.split("|")): datetime.fromisoformat(v) for k,v in d.items()}
    except: return {}
def save_state(state):
    try:
        with open(STATE_FILE,"w") as f:
            json.dump({"|".join(k): v.isoformat() for k,v in state.items()}, f)
    except: pass

def run_once():
    state=load_state()
    df1=get_1m()
    if df1 is None or len(df1)<200: print("No 1m data"); return
    last_t=df1.index[-1]
    now_ny=datetime.now(NY)
    if now_ny.weekday()>=5 or not (dtime(9,30)<=now_ny.time()<=dtime(16,0)):
        print(f"Market closed NY time {now_ny}"); return
    df3=to_3m(df1)
    if df3.index[-1]+timedelta(minutes=2)>last_t: df3=df3.iloc[:-1]
    if len(df3)<60: print("Not enough 3m"); return
    res=analyse(df3)
    today=df1[df1.index.date==last_t.date()]; or_win=today.between_time("09:30","09:44"); after=today.between_time("09:45","16:00")
    or_ready=len(or_win)>0 and len(after)>0
    broke={"CALL":False,"PUT":False}
    if or_ready:
        broke["CALL"]=bool((after["High"]>or_win["High"].max()).any())
        broke["PUT"]=bool((after["Low"]<or_win["Low"].min()).any())
    for side in ("CALL","PUT"):
        agree,candles=res[side]
        if len(agree)<1 or not candles: continue
        todo=["شموع"]
        if broke[side]: todo.append("ORB")
        for strat in todo:
            key=(strat,side)
            if key in state and last_t-state[key]<timedelta(minutes=COOLDOWN_MIN):
                print(f"Cooldown {key}"); continue
            send_telegram(build_msg(strat,side,agree,candles))
            state[key]=last_t
    save_state(state)

if __name__=="__main__":
    run_once()
