"""Build the static Marathi ASR leaderboard dashboard.

Reads provenance-stamped records from spaces/results/*.json (one per model, written by
run_eval.py) and emits a single self-contained spaces/site/index.html. No server, no CDN.

A record is rejected unless it carries checkpoint provenance (path + sha256), an eval date and a
norm_version equal to the frozen one. This is what keeps stale or hand-typed numbers off the board.

    python spaces/build_dashboard.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SITE_DIR = HERE / "site"
NORM_VERSION = "mr-norm-1"

# id -> display metadata. `split` is what is actually scored (see benchmark.yaml).
DATASETS = {
    "respin_mr": dict(label="RESPIN", split="test", style="read", note="Dialect-labelled (D1-D4); 2,170 utts", default=True),
    "kathbath_mr": dict(label="Kathbath", split="valid", style="read", note="Repo ships no test split; valid used", default=True),
    "indicvoices_mr": dict(label="IndicVoices", split="valid", style="spontaneous", note="Repo ships no test split; valid used", default=True),
    "fleurs_mr": dict(label="FLEURS", split="test", style="read", note="google/fleurs mr_in", default=True),
}
UNAVAILABLE = [
    dict(id="vaani_mr", label="Vaani", why="Hub repo exposes train shards only; no held-out split"),
    dict(id="vimarsha_mr", label="Vimarsha", why="ai4bharat/vimarsha not found on the Hub (404)"),
    dict(id="common_voice_mr", label="Common Voice", why="No data files on the Hub mirror; must be fetched from the source"),
    dict(id="springinx_mr", label="SPRING-INX", why="Configured path is a TTS corpus, not the SPRING-INX eval set"),
]
REQUIRED = ("model", "checkpoint", "eval_date", "norm_version", "scores")


def validate(rec: dict, name: str) -> str | None:
    for k in REQUIRED:
        if k not in rec:
            return f"missing '{k}'"
    if rec["norm_version"] != NORM_VERSION:
        return f"norm_version {rec['norm_version']} != {NORM_VERSION}"
    ck = rec["checkpoint"]
    if not (isinstance(ck, dict) and ck.get("path") and ck.get("sha256")):
        return "checkpoint needs path and sha256"
    for ds, m in rec["scores"].items():
        if ds not in DATASETS:
            return f"unknown dataset {ds}"
        if not all(x in m for x in ("wer", "cer", "n_utts")):
            return f"{ds}: needs wer, cer, n_utts"
    return None


def load_records():
    good, bad = [], []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            bad.append((p.name, f"unreadable: {e}"))
            continue
        err = validate(rec, p.name)
        (bad.append((p.name, err)) if err else good.append(rec))
    return good, bad


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Marathi ASR Leaderboard</title>
<style>
:root{--bg:#fafaf9;--panel:#fff;--ink:#1c1917;--mute:#78716c;--line:#e7e5e4;--acc:#4338ca;--good:#15803d;--bad:#b91c1c;--warn:#b45309}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0c0a09;--panel:#1c1917;--ink:#f5f5f4;--mute:#a8a29e;--line:#292524;--acc:#818cf8;--good:#4ade80;--bad:#f87171;--warn:#fbbf24}}
:root[data-theme=dark]{--bg:#0c0a09;--panel:#1c1917;--ink:#f5f5f4;--mute:#a8a29e;--line:#292524;--acc:#818cf8;--good:#4ade80;--bad:#f87171;--warn:#fbbf24}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{margin:0 0 4px;font-size:26px}h2{font-size:18px;margin:24px 0 8px}
.sub{color:var(--mute);margin:0 0 16px}nav{display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin:16px 0}
nav button{background:none;border:0;border-bottom:2px solid transparent;color:var(--mute);padding:8px 12px;font:inherit;cursor:pointer}
nav button.on{color:var(--ink);border-color:var(--acc)}section{display:none}section.on{display:block}
.bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
.bar label{display:inline-flex;gap:6px;align-items:center}input[type=search]{padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);font:inherit}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}th{font-weight:600;color:var(--mute);font-size:13px}tr:last-child td{border-bottom:0}
.up{color:var(--good)}.down{color:var(--bad)}.na{color:var(--mute)}.best{font-weight:700}
.tag{font-size:11px;padding:1px 6px;border-radius:4px;border:1px solid var(--line);color:var(--mute)}.tag.seen{color:var(--warn);border-color:var(--warn)}
.card{border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:12px 16px;margin:8px 0}
.empty{padding:32px;text-align:center;color:var(--mute);border:1px dashed var(--line);border-radius:8px}
code{font:12px ui-monospace,monospace;background:var(--line);padding:1px 5px;border-radius:4px;word-break:break-all}
svg text{fill:var(--ink);font-size:11px}svg .ax{stroke:var(--line)}
</style></head><body><main>
<h1>Marathi ASR Leaderboard</h1>
<p class="sub">WER, lower is better. Every number comes from a fresh offline run of a named checkpoint, scored under <code>mr-norm-1</code>. Built <span id="built"></span>.</p>
<nav id="tabs"></nav>

<section id="t-board"><div class="card" id="secret" style="display:none">RESPIN-only view. Press Alt+R or remove <code>#respin</code> from the URL to leave.</div><div class="bar" id="dsbar"></div>
<div class="bar"><input id="q" type="search" placeholder="Search models"><label><input type="checkbox" id="clean"> Exclude datasets a model trained on</label></div>
<div class="wrap"><table id="board"></table></div>
<p class="sub" id="boardnote"></p></section>

<section id="t-detail"><div id="detail"></div></section>
<section id="t-slices"><div id="slices"></div></section>
<section id="t-pareto"><div id="pareto"></div></section>
<section id="t-prov"><div id="prov"></div></section>
<section id="t-about"><div id="about"></div></section>
</main>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s), esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const DS=D.datasets, M=D.models, f=(x,d=2)=>x==null?'<span class=na>NA</span>':Number(x).toFixed(d);
$('#built').textContent=D.built;
const tabs=[['board','Leaderboard'],['detail','Per-dataset'],['slices','Dialect slices'],['pareto','Size vs WER'],['prov','Provenance'],['about','About']];
$('#tabs').innerHTML=tabs.map(([k,v])=>`<button data-k="${k}">${v}</button>`).join('');
function show(k){document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id=='t-'+k));document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.k==k))}
$('#tabs').onclick=e=>e.target.dataset.k&&show(e.target.dataset.k);show('board');

const defaults=Object.keys(DS).filter(k=>DS[k].default);
let sel=new Set(defaults);
$('#dsbar').innerHTML='<b>Average over:</b>'+Object.keys(DS).map(k=>`<label><input type=checkbox value="${k}" ${sel.has(k)?'checked':''}> ${DS[k].label}</label>`).join('');
$('#dsbar').onchange=e=>{e.target.checked?sel.add(e.target.value):sel.delete(e.target.value);renderBoard()};
$('#q').oninput=renderBoard;$('#clean').onchange=renderBoard;
// hidden switch: open with #respin, or toggle with Alt+R
function respinOnly(on){sel=on?new Set(['respin_mr']):new Set(defaults);document.querySelectorAll('#dsbar input').forEach(i=>i.checked=sel.has(i.value));$('#secret').style.display=on?'block':'none';
 if(on){show('board');history.replaceState(null,'','#respin')}else if(location.hash=='#respin')history.replaceState(null,'',location.pathname+location.search);renderBoard()}
document.addEventListener('keydown',e=>{if(e.altKey&&e.key.toLowerCase()=='r'){e.preventDefault();respinOnly($('#secret').style.display=='none')}});
window.addEventListener('hashchange',()=>respinOnly(location.hash=='#respin'));

function avg(m,cols,clean){const use=cols.filter(c=>!(clean&&(m.trained_on||[]).includes(c)));
 if(!use.length||use.some(c=>!m.scores[c]))return null;return use.reduce((a,c)=>a+m.scores[c].wer,0)/use.length}
function rank(cols,clean){const r=M.map(m=>[m,avg(m,cols,clean)]).filter(x=>x[1]!=null).sort((a,b)=>a[1]-b[1]);const o={};r.forEach(([m],i)=>o[m.model]=i+1);return o}
function renderBoard(){
 if(!M.length){$('#board').outerHTML='<div class=empty id=board>No verified results yet. Run <code>run_eval.py</code> and drop the record into <code>spaces/results/</code>.</div>';return}
 const cols=[...sel],clean=$('#clean').checked,base=rank(defaults,false),cur=rank(cols,clean),q=$('#q').value.toLowerCase();
 let rows=M.filter(m=>cur[m.model]&&m.model.toLowerCase().includes(q)).sort((a,b)=>cur[a.model]-cur[b.model]);
 const best={};cols.forEach(c=>best[c]=Math.min(...M.filter(m=>m.scores[c]).map(m=>m.scores[c].wer)));
 $('#board').innerHTML='<tr><th>Rank</th><th>Rank Δ</th><th>Model</th><th>Avg WER</th><th>Params (active)</th><th>RTFx</th>'+cols.map(c=>`<th>${DS[c].label}</th>`).join('')+'</tr>'+
 rows.map(m=>{const d=base[m.model]!=null?base[m.model]-cur[m.model]:null;
  return `<tr><td>${cur[m.model]}</td><td class="${d>0?'up':d<0?'down':'na'}">${d==null?'–':d>0?'▲'+d:d<0?'▼'+-d:'='}</td><td>${esc(m.display||m.model)}</td><td class=best>${f(avg(m,cols,clean))}</td><td>${m.params_active_m??'?'}M / ${m.params_total_m??'?'}M</td><td>${f(m.rtfx,1)}</td>`+
  cols.map(c=>{const s=m.scores[c],seen=(m.trained_on||[]).includes(c);return `<td class="${s&&s.wer==best[c]?'best':''}">${s?f(s.wer):'<span class=na>NA</span>'}${seen?' <span class="tag seen" title="trained on this dataset">seen</span>':''}</td>`}).join('')+'</tr>'}).join('');
 const hidden=M.length-rows.length;
 $('#boardnote').textContent=(hidden?hidden+' model(s) hidden: missing a score on a selected dataset (not ranked, never zero-filled). ':'')+'Average is the macro mean over selected datasets. "seen" = the model trained on that dataset\'s training data.';
}
renderBoard();
if(location.hash=='#respin')respinOnly(true);

// per-dataset detail
$('#detail').innerHTML=Object.keys(DS).map(k=>{const ms=M.filter(m=>m.scores[k]).sort((a,b)=>a.scores[k].wer-b.scores[k].wer);
 return `<h2>${DS[k].label} <span class=tag>${DS[k].split}</span> <span class=tag>${DS[k].style}</span></h2><p class=sub>${DS[k].note}</p>`+(ms.length?`<div class=wrap><table><tr><th>#</th><th></th><th>Model</th><th>WER</th><th>95% CI</th><th>CER</th><th>Utts</th><th>Hours</th></tr>`+
 ms.map((m,i)=>{const s=m.scores[k];return `<tr><td>${i+1}</td><td></td><td>${esc(m.display||m.model)}</td><td class=best>${f(s.wer)}</td><td>${s.wer_ci95?f(s.wer_ci95[0],1)+' – '+f(s.wer_ci95[1],1):'<span class=na>NA</span>'}</td><td>${f(s.cer)}</td><td>${s.n_utts}</td><td>${f(s.hours,2)}</td></tr>`}).join('')+'</table></div>':'<div class=empty>Not evaluated yet.</div>')}).join('');

// slices
(function(){const out=[];Object.keys(DS).forEach(k=>{const kinds={};M.forEach(m=>Object.entries((m.slices||{})[k]||{}).forEach(([t,g])=>{kinds[t]=kinds[t]||new Set();Object.keys(g).forEach(x=>kinds[t].add(x))}));
 Object.entries(kinds).forEach(([t,gs])=>{gs=[...gs].sort();out.push(`<h2>${DS[k].label}: by ${t}</h2><div class=wrap><table><tr><th></th><th></th><th>Model</th>`+gs.map(g=>`<th>${esc(g)}</th>`).join('')+'</tr>'+M.filter(m=>(m.slices||{})[k]&&m.slices[k][t]).map(m=>`<tr><td></td><td></td><td>${esc(m.display||m.model)}</td>`+gs.map(g=>{const c=m.slices[k][t][g];return `<td>${c?f(c.wer)+' <span class=na>n='+c.n+'</span>':'<span class=na>NA</span>'}</td>`}).join('')+'</tr>').join('')+'</table></div>')})});
 $('#slices').innerHTML=out.join('')||'<div class=empty>No slice data yet.</div>'})();

// pareto: active params vs avg WER
(function(){const pts=M.map(m=>({m,x:m.params_active_m,y:avg(m,defaults,false)})).filter(p=>p.x&&p.y!=null);
 if(!pts.length){$('#pareto').innerHTML='<div class=empty>No models with full default coverage yet.</div>';return}
 const W=760,H=380,P=48,xs=pts.map(p=>Math.log10(p.x)),ys=pts.map(p=>p.y),x0=Math.min(...xs)-.1,x1=Math.max(...xs)+.1,y0=0,y1=Math.max(...ys)*1.1;
 const X=v=>P+(Math.log10(v)-x0)/(x1-x0)*(W-P-16),Y=v=>H-P-(v-y0)/(y1-y0)*(H-P-16);
 $('#pareto').innerHTML=`<h2>Active parameters vs average WER</h2><p class=sub>Default datasets only. Bottom-left is better.</p><svg viewBox="0 0 ${W} ${H}" width="100%"><line class=ax x1=${P} y1=${H-P} x2=${W-16} y2=${H-P}/><line class=ax x1=${P} y1=16 x2=${P} y2=${H-P}/>`+
 pts.map(p=>`<circle cx=${X(p.x)} cy=${Y(p.y)} r=6 fill="var(--acc)"><title>${esc(p.m.model)}: ${p.y.toFixed(2)} WER, ${p.x}M active</title></circle><text x=${X(p.x)+9} y=${Y(p.y)+4}>${esc(p.m.display||p.m.model)}</text>`).join('')+
 `<text x=${W/2} y=${H-12} text-anchor=middle>active parameters (M, log)</text><text x=14 y=${H/2} transform="rotate(-90 14 ${H/2})" text-anchor=middle>average WER</text></svg>`})();

// provenance
$('#prov').innerHTML=(M.length?M.map(m=>`<div class=card><b>${esc(m.display||m.model)}</b><br>Checkpoint: <code>${esc(m.checkpoint.path)}</code><br>SHA-256: <code>${esc(m.checkpoint.sha256)}</code><br>Evaluated ${esc(m.eval_date)} on ${esc(m.hardware||'unknown hardware')} · norm <code>${esc(m.norm_version)}</code> · language forced: ${m.language_forced?'yes':'no / n.a.'}<br>Trained on: ${(m.trained_on||[]).length?m.trained_on.map(esc).join(', '):'<span class=na>unknown</span>'}</div>`).join(''):'<div class=empty>No records.</div>')+
 (D.rejected.length?'<h2>Rejected records</h2>'+D.rejected.map(r=>`<div class=card><code>${esc(r[0])}</code> ${esc(r[1])}</div>`).join(''):'');

$('#about').innerHTML=`<h2>Method</h2><div class=card>Offline evaluation on a fixed GPU, one harness for every model. WER and CER after <code>mr-norm-1</code> normalisation of both reference and hypothesis. Average WER is the macro mean over selected datasets. Models missing a selected dataset are not ranked. 95% intervals are bootstrap over utterances. WER above 100 is legitimate and not clipped.</div>
<h2>Datasets not on the board</h2>`+D.unavailable.map(u=>`<div class=card><b>${esc(u.label)}</b>: ${esc(u.why)}</div>`).join('')+
`<h2>Caveats</h2><div class=card>Kathbath and IndicVoices are scored on their <code>valid</code> split because the Hub repos ship no test split. Models listed as "seen" were trained on that dataset's training data, so compare them with the exclusion toggle on. This page is a static snapshot; it is not a live demo.</div>`;
</script></body></html>
"""


def main():
    good, bad = load_records()
    data = dict(
        built=__import__("datetime").date.today().isoformat(),
        datasets=DATASETS,
        unavailable=UNAVAILABLE,
        models=good,
        rejected=bad,
    )
    SITE_DIR.mkdir(exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print(f"wrote {out}: {len(good)} model(s), {len(bad)} rejected")
    for n, why in bad:
        print(f"  rejected {n}: {why}", file=sys.stderr)


if __name__ == "__main__":
    main()
