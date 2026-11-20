# trial ブランチ（試運転用）

upstream `main`（`d2a29ef`）に、次の 6 本をマージした実行用のブランチ。upstream には無い。

| ブランチ | 中身 |
| --- | --- |
| `fix/issue-5-undefined-methods` | 定義の無いメソッド 3 件（upstream #5） |
| `fix/issue-8-bullish-defaults` | 取得失敗・実績なしのときの既定値（#8） |
| `fix/issue-6-key-mismatch` | フィードとエージェントのキー名、RS と F-Score の計算（#6） |
| `fix/issue-7-timesfm-label` | 予測のバックエンドに合わせた表示（#7） |
| `fix/issue-9-ibkr-bracket` | IBKR の戻り値と文書（#9） |
| `feat/webull-intents` | Webull 向けに「発注の意図」を書き出すブローカー |

## 準備

Python 3.11 を使う（upstream の CI は 3.10 / 3.11）。

```bash
uv venv --python 3.11 .venv
```

```bash
uv pip install --python .venv/bin/python -e ".[test]" -r requirements.txt matplotlib
```

`matplotlib` は upstream の `requirements.txt` に無いが、ティアシートが読み込む。

## テスト

```bash
APPDATA="$PWD/.trial/appdata" .venv/bin/python -m pytest tests/ -q
```

`APPDATA` を指定しないと、台帳などがカレントディレクトリ相対の `LuminaAI/` に作られる。

## 1 サイクルだけ回す

Webull の API は呼ばない。データは yfinance と FRED から取る。LLM の API キーが無いときは、合議は縮退モード（決定的な規則）で動く。

環境変数（`.env` に書くか、シェルで `export` する）:

| 変数 | 値の例 | 意味 |
| --- | --- | --- |
| `APPDATA` | `$PWD/.trial/appdata` | 状態ファイルの置き場 |
| `LUMINA_BROKER` | `webull_intents` | 既定は `paper` |
| `LUMINA_DB_PATH` | `$PWD/.trial/webull.db` | 台帳（SQLite） |
| `LUMINA_WEBULL_INTENT_DIR` | `$PWD/.trial/intents` | 意図と結果の JSONL の置き場 |
| `LUMINA_WEBULL_INITIAL_USD` | `3200` | 新しい台帳を米ドルだけの元手にする |
| `LUMINA_WEBULL_FILL_MODE` | `paper` | `paper` = 参考価格ですぐ記帳 / `await` = 執行機の結果を待つ |
| `LUMINA_NYSE_CALENDAR_DIR` | `~/webull-api-jp/refdata` | 休業日と半日立会の表。無ければ平日だけで判定 |

```bash
.venv/bin/python scripts/trial_cycle.py run
```

立会の外でも、スクリーニングと合議までを通す（発注はブローカーが別に判定して拒む）:

```bash
.venv/bin/python scripts/trial_cycle.py run --ignore-hours
```

台帳と意図の数を見る:

```bash
.venv/bin/python scripts/trial_cycle.py status
```

## ダッシュボード

```bash
.venv/bin/streamlit run app.py
```

## Webull との関係

- `WebullIntentBroker` は意図を `intents-<ET の日付>.jsonl` に書くだけで、Webull には何も送らない。
- 意図を読んで関門を通し、発注するのは、別のリポジトリの執行機（`webull-api-jp` の `usm`）の仕事。
- `await` のときは、執行機が `results-<ET の日付>.jsonl` に書く結果を待ち、約定した分だけを約定値で台帳に記帳する。
