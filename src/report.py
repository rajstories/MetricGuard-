"""
MetricGuard - Report Generator
Runs the full pipeline and writes results to output/results.json,
which the dashboard reads. Also prints headline KPIs for your resume.
"""
import json
from pathlib import Path
from engine import run_analysis, load_metrics
from genai import load_glossary, resolve_conflict

OUT = Path(__file__).parent.parent / "output"
OUT.mkdir(exist_ok=True)


def write_data_js(payload):
    """Write output/data.js so dashboard.html can load results without a server."""
    js = "const MG_DATA = " + json.dumps(payload, ensure_ascii=False) + ";"
    with open(OUT / "data.js", "w") as f:
        f.write(js)


def main(use_llm=True):
    metrics, results = run_analysis()
    glossary = load_glossary()

    enriched = []
    for r in results:
        item = {
            "names": r["names"],
            "teams": r["teams"],
            "metric_ids": r["metric_ids"],
            "conflicts": r["conflicts"],
            "trust_risk": r["trust_risk"],
            "avg_similarity": round(r["avg_similarity"], 3),
            "definitions": [
                {"team": m["team"], "name": m["metric_name"],
                 "description": m["description"], "sql": m["sql"]}
                for m in r["metrics"]
            ],
        }
        if use_llm:
            res = resolve_conflict(r, glossary)
            item["recommended_concept"] = res["concept"]
            item["glossary_owner"] = res["retrieved_glossary"]["owner"]
            item["retrieval_similarity"] = res["retrieval_similarity"]
            item["recommendation"] = res["recommendation"]
        enriched.append(item)

    # headline KPIs (these become your resume numbers)
    total_defs = len(metrics)
    conflicting_defs = sum(len(r["metric_ids"]) for r in results)
    teams_affected = len({t for r in results for t in r["teams"]})
    kpis = {
        "total_definitions_scanned": total_defs,
        "conflicting_definitions_found": conflicting_defs,
        "conflict_groups": len(results),
        "teams_affected": teams_affected,
        "pct_definitions_in_conflict": round(100 * conflicting_defs / total_defs, 1),
        "highest_trust_risk": max((r["trust_risk"] for r in results), default=0),
    }

    payload = {"kpis": kpis, "conflicts": enriched}
    with open(OUT / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    write_data_js(payload)

    print("HEADLINE KPIs (use these on your resume):")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {OUT / 'results.json'} and {OUT / 'data.js'}")
    return payload


if __name__ == "__main__":
    main()
