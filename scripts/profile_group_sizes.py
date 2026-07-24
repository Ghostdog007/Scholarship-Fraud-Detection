"""K_CAP profiling query (open decision #1, TECHNICAL_REFERENCE_AND_SCALING.md
§12.4 / §15). Derives the hub-cap threshold and the frequency ceiling from the
OBSERVED group-size distribution of each identity relation — never a
hand-picked number (hard stop 1).

Reusable: rerun this against the real 3.5M ingest when it lands
(`python -m scripts.profile_group_sizes`); today it reports on whatever is
currently in the `applications`/`identity_keys` tables (merged batches only,
same population `edge_groups()` builds the graph from).

Reports, per relation:
  - group count, total "shared" applications (group size >= 2)
  - group-size percentiles (p50/p90/p95/p99/p99.9/max)
  - candidate ceiling (99.9th percentile of group size, per config_v3 comment)
  - edge-count impact: raw all-pairs cliques vs. capped (star above k_cap,
    dropped above ceiling) at a few candidate k_cap values, so the tradeoff
    is visible before a value is chosen.
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from src.db.connection import get_connection  # noqa: E402

RELATIONS = {
    "shares_mobile":      "mobile_no",
    "shares_ip":          "ip_address",
    "shares_father_name": "father_name_norm",
    "shares_mother_name": "mother_name_norm",
    "shares_pincode":     "pincode",
}

CANDIDATE_K_CAPS = [10, 20, 50, 100]


def clique_edges(k: int) -> int:
    return k * (k - 1) // 2


def capped_edges(k: int, k_cap: int, ceiling: int) -> int:
    if k > ceiling:
        return 0
    if k > k_cap:
        return k - 1   # star: k-1 edges to the hub
    return clique_edges(k)


def profile_relation(rel: str, col: str, conn) -> dict:
    rows = conn.execute(
        f"SELECT COUNT(*) FROM ("
        f"  SELECT ik.{col} FROM identity_keys ik"
        f"  JOIN applications a ON a.application_id = ik.application_id"
        f"  JOIN batches b ON b.batch_id = a.batch_id"
        f"  WHERE b.status = 'merged' AND ik.{col} IS NOT NULL"
        f"  GROUP BY ik.{col} HAVING COUNT(*) >= 2"
        f") g"
    ).fetchone()
    n_groups = rows[0]

    if n_groups == 0:
        return {"rel": rel, "n_groups": 0}

    sizes = conn.execute(
        f"SELECT COUNT(*) AS sz FROM identity_keys ik"
        f" JOIN applications a ON a.application_id = ik.application_id"
        f" JOIN batches b ON b.batch_id = a.batch_id"
        f" WHERE b.status = 'merged' AND ik.{col} IS NOT NULL"
        f" GROUP BY ik.{col} HAVING COUNT(*) >= 2"
    ).fetchall()
    sizes = np.array([r[0] for r in sizes], dtype=np.int64)

    pcts = {p: float(np.percentile(sizes, p)) for p in (50, 90, 95, 99, 99.9)}
    ceiling = max(int(np.percentile(sizes, 99.9)), 2)

    raw_edges = int(sum(clique_edges(int(k)) for k in sizes))
    capped_at = {
        k_cap: int(sum(capped_edges(int(k), k_cap, ceiling) for k in sizes))
        for k_cap in CANDIDATE_K_CAPS
    }

    return {
        "rel": rel, "n_groups": n_groups, "n_shared_apps": int(sizes.sum()),
        "min": int(sizes.min()), "max": int(sizes.max()), "mean": float(sizes.mean()),
        "pcts": pcts, "candidate_ceiling_p999": ceiling,
        "raw_clique_edges": raw_edges, "capped_edges": capped_at,
    }


def main() -> None:
    with get_connection() as conn:
        n_apps = conn.execute(
            "SELECT COUNT(*) FROM applications a JOIN batches b"
            " ON b.batch_id = a.batch_id WHERE b.status = 'merged'"
        ).fetchone()[0]
        print(f"[profile] merged population: {n_apps:,} applications\n")

        results = []
        for rel, col in RELATIONS.items():
            r = profile_relation(rel, col, conn)
            results.append(r)
            if r["n_groups"] == 0:
                print(f"[profile] {rel}: no shared groups (all values unique or null)\n")
                continue
            print(f"[profile] {rel}  (column: {col})")
            print(f"  groups (size>=2): {r['n_groups']:,}   "
                  f"applications in a shared group: {r['n_shared_apps']:,} "
                  f"({100*r['n_shared_apps']/n_apps:.1f}% of population)")
            print(f"  group size: min={r['min']} mean={r['mean']:.2f} max={r['max']}")
            p = r["pcts"]
            print(f"  percentiles: p50={p[50]:.0f} p90={p[90]:.0f} p95={p[95]:.0f} "
                  f"p99={p[99]:.0f} p99.9={p[99.9]:.0f}")
            print(f"  candidate ceiling (p99.9): {r['candidate_ceiling_p999']}")
            print(f"  raw all-pairs clique edges: {r['raw_clique_edges']:,}")
            for k_cap, e in r["capped_edges"].items():
                reduction = (1 - e / r["raw_clique_edges"]) * 100 if r["raw_clique_edges"] else 0
                print(f"    k_cap={k_cap:>4}: {e:,} edges ({reduction:.1f}% reduction)")
            print()

    print("=" * 70)
    print("[profile] NOTE: this population is whatever is currently ingested —")
    print("check the 'merged population' line above against your intended scale.")
    print("Candidate K_CAP/ceiling values here are a methodology dry run, not a")
    print("production decision, unless the population above is the real target.")


if __name__ == "__main__":
    main()
