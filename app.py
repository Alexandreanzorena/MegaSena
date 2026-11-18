"""
Mega-Sena Lab - aplicativo de estudo (Flask + scikit-learn)
Importa o xlsx de resultados, treina modelos, valida contra o acaso,
gera previsões para estudo e faz backtest de ROI.

AVISO: a Mega-Sena é um sorteio aleatório e independente. Este app é
educacional: nenhum modelo supera o acaso de forma consistente.
"""
import io, threading, uuid, traceback, warnings
from math import comb
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings("ignore")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

STATE = {"df": None, "Y": None, "F": None, "prem": None, "probs": None,
         "model_info": None, "last_games": None, "sums": None}
JOBS = {}
WARMUP = 300      # concursos iniciais usados só para aquecer as features
WIN = 20          # janela da rede neural (nº de concursos anteriores)
FEATS = ["atraso", "freq10", "freq30", "freq100", "freq300", "freq_total",
         "atraso/gap_medio", "dezena", "par", "linha", "coluna", "quadrante",
         "freq_quadrante10", "falta_no_ciclo", "tam_ciclo", "faltam_ciclo", "saiu_anterior"]
ROWS = np.arange(60) // 10
COLS = np.arange(60) % 10
QUAD = (ROWS >= 3).astype(int) * 2 + (COLS >= 5).astype(int)   # 4 quadrantes do volante (6x10)


# ------------------------------------------------------------------ dados
def money(v):
    if isinstance(v, (int, float, np.number)):
        return 0.0 if pd.isna(v) else float(v)
    s = str(v).replace("R$", "").replace(".", "").replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_xlsx(raw):
    xl = pd.ExcelFile(io.BytesIO(raw))
    for sheet in xl.sheet_names:
        d = xl.parse(sheet)
        cols = {str(c).strip().lower(): c for c in d.columns}
        for pattern in ("bola{}", "dezena{}", "d{}"):
            names = [pattern.format(i) for i in range(1, 7)]
            if all(n in cols for n in names):
                break
        else:
            continue
        balls = d[[cols[n] for n in names]].apply(pd.to_numeric, errors="coerce")
        ok = balls.notna().all(axis=1)
        d, balls = d[ok].copy(), balls[ok].astype(int)
        v = balls.values
        valid = ((v >= 1) & (v <= 60)).all(axis=1) & (np.sort(v, 1)[:, 1:] != np.sort(v, 1)[:, :-1]).all(axis=1)
        d, balls = d[valid].copy(), balls[valid]
        out = pd.DataFrame({f"b{i+1}": balls.iloc[:, i].values for i in range(6)})
        ccol = next((c for k, c in cols.items() if k.startswith("concurso")), None)
        dcol = next((c for k, c in cols.items() if k.startswith("data")), None)
        out["concurso"] = pd.to_numeric(d[ccol], errors="coerce").values if ccol else np.arange(1, len(d) + 1)
        out["data"] = pd.to_datetime(d[dcol], dayfirst=True, errors="coerce").values if dcol else pd.NaT
        for q in (4, 5, 6):
            c = next((c for k, c in cols.items() if k.startswith(f"rateio {q}")), None)
            out[f"rateio{q}"] = d[c].map(money).values if c else 0.0
        out = out.sort_values("concurso").reset_index(drop=True)
        return out, sheet
    raise ValueError("Não encontrei as colunas Bola1..Bola6 (ou Dezena1..Dezena6) em nenhuma aba.")


def build_matrix(df):
    Y = np.zeros((len(df), 60), dtype=np.int64)
    for i, row in enumerate(df[[f"b{i}" for i in range(1, 7)]].values):
        Y[i, row - 1] = 1
    return Y


def build_features(Y):
    """Features por (concurso, dezena). A linha t usa SOMENTE concursos < t.
    Retorna N+1 linhas: a última é o PRÓXIMO concurso (ainda não sorteado)."""
    N = len(Y)
    F = np.zeros((N + 1, 60, len(FEATS)), dtype=np.float32)
    delay = np.zeros(60); cum = np.zeros(60)
    seen = np.zeros(60, bool); clen = 0
    csum = np.vstack([np.zeros((1, 60)), np.cumsum(Y, 0)])
    nums = np.arange(1, 61)
    for t in range(N + 1):
        f = lambda w: (csum[t] - csum[max(t - w, 0)]) / w
        meangap = np.where(cum > 0, t / np.maximum(cum, 1), t + 1)
        rec = csum[t] - csum[max(t - 10, 0)]
        qc = np.array([rec[QUAD == q].sum() / 60 for q in range(4)])
        missing = (~seen).astype(float)
        F[t, :, 0] = delay; F[t, :, 1] = f(10); F[t, :, 2] = f(30); F[t, :, 3] = f(100)
        F[t, :, 4] = f(300); F[t, :, 5] = csum[t] / max(t, 1)
        F[t, :, 6] = delay / np.maximum(meangap, 1)
        F[t, :, 7] = nums; F[t, :, 8] = nums % 2; F[t, :, 9] = ROWS; F[t, :, 10] = COLS
        F[t, :, 11] = QUAD; F[t, :, 12] = qc[QUAD]
        F[t, :, 13] = missing; F[t, :, 14] = clen; F[t, :, 15] = missing.sum()
        F[t, :, 16] = Y[t - 1] if t > 0 else 0
        if t < N:
            drawn = Y[t] == 1
            delay = np.where(drawn, 0, delay + 1); cum += drawn
            seen |= drawn; clen += 1
            if seen.all():
                seen[:] = False; clen = 0
    return F


def windows(Y, a, b):
    return np.array([Y[t - WIN:t].reshape(-1) for t in range(a, b)], dtype=np.float32)


# ------------------------------------------------------------- modelagem
def fit_predict(kind, F, Y, train_end, pred_a, pred_b, n_trees):
    """Treina em [WARMUP, train_end) e prevê os concursos [pred_a, pred_b). Retorna (P, importâncias)."""
    if kind == "rf":
        X = F[WARMUP:train_end].reshape(-1, F.shape[2])
        y = Y[WARMUP:train_end].reshape(-1)
        m = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=200, max_depth=8,
                                   n_jobs=-1, random_state=0).fit(X[::2], y[::2])
        P = m.predict_proba(F[pred_a:pred_b].reshape(-1, F.shape[2]))[:, 1].reshape(-1, 60)
        return P, m.feature_importances_
    m = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=80,
                      early_stopping=True, random_state=0)
    m.fit(windows(Y, WARMUP + WIN, train_end), Y[WARMUP + WIN:train_end])
    return m.predict_proba(windows(Y, pred_a, pred_b)), None


def evaluate(P, Yt, rng):
    y = Yt.reshape(-1); p = np.clip(P.reshape(-1), 1e-6, 1 - 1e-6)
    top6 = np.argsort(-P, 1)[:, :6]
    hits = np.take_along_axis(Yt, top6, 1).sum(1)
    perm = [roc_auc_score(y, np.clip(P[rng.permutation(len(P))].reshape(-1), 1e-6, 1 - 1e-6)) for _ in range(100)]
    tt = stats.ttest_1samp(hits, 0.6)
    auc = roc_auc_score(y, p)
    return {"auc": float(auc), "auc_perm_p95": float(np.percentile(perm, 95)),
            "acertos_top6": float(hits.mean()), "acertos_esperado": 0.6,
            "p_valor": float(tt.pvalue), "n_teste": int(len(P)),
            "dist_acertos": {int(k): int((hits == k).sum()) for k in range(7)},
            "sinal": bool(auc > np.percentile(perm, 95) and tt.pvalue < 0.05)}


# ------------------------------------------------------------- geração
def sample_games(rng, probs, n, k, smin, smax, pmin, pmax, unique=True, max_try=300000):
    p = np.power(np.maximum(probs, 1e-9), k); p /= p.sum()
    seen, out, tries = set(), [], 0
    while len(out) < n and tries < max_try:
        tries += 1
        g = tuple(sorted((rng.choice(60, 6, replace=False, p=p) + 1).tolist()))
        if unique and g in seen:
            continue
        s = sum(g); par = sum(1 for x in g if x % 2 == 0)
        if smin <= s <= smax and pmin <= par <= pmax:
            seen.add(g); out.append(g)
    return out


def sum_limits(sums, lo_pct, hi_pct):
    return float(np.percentile(sums, lo_pct)), float(np.percentile(sums, hi_pct))


def teste_acertos(dist):
    """Compara acertos observados com o esperado ao acaso (hipergeométrica) para cada faixa."""
    tot = comb(60, 6); n = int(sum(dist))
    p = {k: comb(6, k) * comb(54, 6 - k) / tot for k in range(7)}
    p4 = p[4] + p[5] + p[6]; obs4 = int(sum(dist[4:]))
    return {"n_jogos": n, "esperado": [float(n * p[k]) for k in range(7)], "observado": [int(x) for x in dist],
            "obs_4mais": obs4, "esp_4mais": float(n * p4),
            "p_valor_4mais": float(stats.binom.sf(obs4 - 1, n, p4)) if obs4 > 0 else 1.0}


# ------------------------------------------------------------------ jobs
def run_job(fn, params):
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"status": "running", "progress": 0.0, "msg": "Iniciando...", "result": None}

    def target():
        try:
            JOBS[jid]["result"] = fn(jid, params)
            JOBS[jid].update(status="done", progress=1.0, msg="Concluído")
        except Exception as e:
            traceback.print_exc()
            JOBS[jid].update(status="error", msg=str(e))
    threading.Thread(target=target, daemon=True).start()
    return jid


def prog(jid, p, msg):
    JOBS[jid]["progress"] = float(p); JOBS[jid]["msg"] = msg


def train_job(jid, prm):
    Y, F = STATE["Y"], STATE["F"]
    N = len(Y); kind = prm["modelo"]; trees = int(prm["arvores"])
    H = int(min(prm["holdout"], N // 3))
    rng = np.random.default_rng(1)
    prog(jid, 0.1, "Validando em concursos que o modelo nunca viu...")
    P, _ = fit_predict(kind, F, Y, N - H, N - H, N, trees)
    ev = evaluate(P, Y[N - H:], rng)
    prog(jid, 0.6, "Treinando o modelo final com todos os concursos...")
    Pn, imp = fit_predict(kind, F, Y, N, N, N + 1, trees)
    probs = Pn[0]
    STATE["probs"] = probs
    info = {"modelo": kind, "arvores": trees, "holdout": H, "avaliacao": ev,
            "probs": [float(x) for x in probs],
            "importancia": None if imp is None else {n: float(v) for n, v in zip(FEATS, imp)}}
    STATE["model_info"] = info
    return info


def backtest_job(jid, prm):
    Y, F, df, prem = STATE["Y"], STATE["F"], STATE["df"], STATE["prem"]
    N = len(Y)
    if df["data"].notna().all():
        start = df["data"].max() - pd.DateOffset(years=int(prm["anos"]))
        T0 = int(np.argmax((df["data"] > start).values))
    else:
        T0 = N - int(prm["anos"]) * 130
    T0 = max(T0, WARMUP + WIN + 50)
    G = int(prm["jogos"]); price = float(prm["preco"]); refit = 100
    smin, smax = sum_limits(STATE["sums"][:T0], prm["soma_min"], prm["soma_max"])
    pmin, pmax = int(prm["par_min"]), int(prm["par_max"])
    rng = np.random.default_rng(int(prm.get("semente", 7)))
    idx = list(range(T0, N)); n_t = len(idx)
    games_model = np.zeros((n_t, G, 6), int); games_rf = None
    for s in range(T0, N, refit):
        e = min(s + refit, N)
        prog(jid, 0.05 + 0.6 * (s - T0) / (N - T0), f"Retreinando e simulando concursos {s}-{e}...")
        P, _ = fit_predict(prm["modelo"], F, Y, s, s, e, 40)
        for t in range(s, e):
            gs = sample_games(rng, P[t - s], G, prm["intensidade"], smin, smax, pmin, pmax, unique=False)
            while len(gs) < G:
                gs.append(tuple(sorted((rng.choice(60, 6, replace=False) + 1).tolist())))
            games_model[t - T0] = np.array(gs)
    prog(jid, 0.7, "Simulando estratégias aleatórias para comparar...")
    Yt = Y[T0:]; premt = prem[T0:]

    def score(games):  # games: (n_t, G, 6) com dezenas 1..60
        hits = np.take_along_axis(Yt[:, None, :].repeat(games.shape[1], 1), games - 1, 2).sum(2)
        ret = np.where(hits == 4, premt[:, 0:1], 0) + np.where(hits == 5, premt[:, 1:2], 0) + np.where(hits == 6, premt[:, 2:3], 0)
        return hits, ret

    def pack(hits, ret):
        inv = hits.size * price
        per = ret.sum(1) - G * price
        step = max(1, len(per) // 200)
        return {"investido": float(inv), "retorno": float(ret.sum()),
                "roi": float(100 * (ret.sum() - inv) / inv),
                "dist": [int((hits == k).sum()) for k in range(7)],
                "premios_altos": int((hits >= 5).sum()),
                "curva": [float(x) for x in np.cumsum(per)[::step]]}
    h, r = score(games_model); model = pack(h, r)
    rois = []
    for _ in range(40):
        rg = np.argsort(rng.random((n_t * G, 60)), 1)[:, :6].reshape(n_t, G, 6) + 1
        hh, rr = score(rg); rois.append(100 * (rr.sum() - hh.size * price) / (hh.size * price))
    rg = np.argsort(rng.random((n_t * G, 60)), 1)[:, :6].reshape(n_t, G, 6) + 1
    rand = pack(*score(rg))
    rois = np.array(rois)
    comb_p = comb(60, 6)
    return {"inicio": str(df["data"].iloc[T0].date()) if pd.notna(df["data"].iloc[T0]) else f"concurso {df['concurso'].iloc[T0]}",
            "n_concursos": n_t, "jogos_por_concurso": G, "preco": price,
            "modelo": model, "aleatorio": rand,
            "aleatorio_dist": {"p5": float(np.percentile(rois, 5)), "p50": float(np.percentile(rois, 50)),
                               "p95": float(np.percentile(rois, 95)), "min": float(rois.min()), "max": float(rois.max())},
            "teste_acertos": teste_acertos(model["dist"]),
            "prob_quina_por_jogo": comb(6, 5) * 54 / comb_p,
            "tem_premios": bool(premt.sum() > 0)}


# ------------------------------------------------------------------ rotas
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("arquivo")
    if not f:
        return jsonify(erro="Envie um arquivo .xlsx"), 400
    try:
        df, sheet = parse_xlsx(f.read())
    except Exception as e:
        return jsonify(erro=str(e)), 400
    if len(df) < 800:
        return jsonify(erro=f"Poucos concursos válidos ({len(df)}). São necessários pelo menos 800."), 400
    STATE.update(df=df, Y=build_matrix(df), probs=None, model_info=None, last_games=None)
    STATE["F"] = build_features(STATE["Y"])
    STATE["sums"] = df[[f"b{i}" for i in range(1, 7)]].sum(1).values
    STATE["prem"] = df[["rateio4", "rateio5", "rateio6"]].values
    last = df.iloc[-1]
    return jsonify(aba=sheet, concursos=int(len(df)),
                   primeiro=int(df.concurso.iloc[0]), ultimo=int(last.concurso),
                   data_inicio=None if pd.isna(df.data.iloc[0]) else str(df.data.iloc[0].date()),
                   data_fim=None if pd.isna(last.data) else str(last.data.date()),
                   ultimo_sorteio=[int(last[f"b{i}"]) for i in range(1, 7)],
                   premios=bool(df[["rateio4", "rateio5", "rateio6"]].values.sum() > 0),
                   soma_p10=float(np.percentile(STATE["sums"], 10)), soma_p90=float(np.percentile(STATE["sums"], 90)))


def need_data():
    if STATE["Y"] is None:
        return jsonify(erro="Importe o arquivo xlsx primeiro."), 400


@app.route("/api/train", methods=["POST"])
def train():
    if (r := need_data()): return r
    p = request.get_json(force=True)
    p = {"modelo": p.get("modelo", "rf") if p.get("modelo") in ("rf", "mlp") else "rf",
         "arvores": min(max(int(p.get("arvores", 100)), 10), 400),
         "holdout": min(max(int(p.get("holdout", 500)), 100), 1500)}
    return jsonify(job=run_job(train_job, p))


@app.route("/api/backtest", methods=["POST"])
def backtest():
    if (r := need_data()): return r
    p = request.get_json(force=True)
    p = {"modelo": "mlp" if p.get("modelo") == "mlp" else "rf",
         "anos": min(max(int(p.get("anos", 5)), 1), 15),
         "jogos": min(max(int(p.get("jogos", 5)), 1), 30),
         "preco": max(float(p.get("preco", 6.0)), 0.5),
         "intensidade": min(max(float(p.get("intensidade", 1)), 0), 30),
         "semente": int(p.get("semente", 7)),
         "soma_min": float(p.get("soma_min", 10)), "soma_max": float(p.get("soma_max", 90)),
         "par_min": int(p.get("par_min", 2)), "par_max": int(p.get("par_max", 4))}
    return jsonify(job=run_job(backtest_job, p))


@app.route("/api/job/<jid>")
def job(jid):
    j = JOBS.get(jid)
    return (jsonify(j) if j else (jsonify(erro="job não encontrado"), 404))


@app.route("/api/predict", methods=["POST"])
def predict():
    if (r := need_data()): return r
    p = request.get_json(force=True)
    use_model = STATE["probs"] is not None and not p.get("aleatorio")
    probs = STATE["probs"] if use_model else np.full(60, 0.1)
    n = min(max(int(p.get("quantidade", 10)), 1), 1000)
    k = min(max(float(p.get("intensidade", 1)), 0), 30)
    smin, smax = sum_limits(STATE["sums"], float(p.get("soma_min", 10)), float(p.get("soma_max", 90)))
    pmin, pmax = int(p.get("par_min", 2)), int(p.get("par_max", 4))
    seed = p.get("semente")
    rng = np.random.default_rng(int(seed) if str(seed or "").strip() != "" else None)
    games = sample_games(rng, probs, n, k if use_model else 0, smin, smax, pmin, pmax)
    out = []
    for g in games:
        quad = [0, 0, 0, 0]
        for x in g: quad[QUAD[x - 1]] += 1
        out.append({"dezenas": list(g), "soma": int(sum(g)), "pares": int(sum(1 for x in g if x % 2 == 0)),
                    "quadrantes": quad, "score": float(sum(probs[x - 1] for x in g))})
    STATE["last_games"] = out
    return jsonify(jogos=out, solicitados=n, gerados=len(out), usou_modelo=bool(use_model),
                   faixa_soma=[smin, smax])


@app.route("/api/export.xlsx")
def export():
    games = STATE.get("last_games")
    if not games:
        return jsonify(erro="Gere previsões primeiro."), 400
    rows = [{"Jogo": i + 1, **{f"D{j+1}": d for j, d in enumerate(g["dezenas"])}, "Soma": g["soma"],
             "Pares": g["pares"], "Quadrantes (SE,SD,IE,ID)": "-".join(map(str, g["quadrantes"])),
             "Score do modelo": round(g["score"], 4)} for i, g in enumerate(games)]
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, sheet_name="Previsoes")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="previsoes_megasena.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    import webbrowser
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    app.run(host="0.0.0.0", port=8000, debug=False)
