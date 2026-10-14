#!/usr/bin/env python3
"""
Standalone diagnostic for the retrieval + graph-construction stage of the
AVeriTeC pipeline -- runs independently of models5.ipynb (no GAT training,
no batching, no notebook needed) so you can check this one stage quickly.

Run from the project root (the directory that contains `pipeline/` and
`data/`), same as models5.ipynb expects:

    python test_retrieval_pipeline.py --n 10
    python test_retrieval_pipeline.py --n 10 --retrieval-only   # skips graph
                                                                 # construction
                                                                 # (NLI/DBSCAN
                                                                 # per claim)
    python test_retrieval_pipeline.py --claim "some specific claim text" --url "https://..."

Note on --retrieval-only: it skips calling augment_with_cross_source() (the
per-claim NLI + DBSCAN work), which is the bulk of the per-claim cost once
you're testing many claims. It does NOT skip the one-time model loading at
startup (spaCy, the sentence encoder, the NLI cross-encoder all load as soon
as anything in `pipeline` is imported, since `pipeline/config.py` loads them
unconditionally at import time) -- that startup cost happens either way.

What this checks, in order, and why each step exists:

  0. Which DuckDuckGo package actually got imported (`ddgs` vs the
     deprecated `duckduckgo_search`) -- if you're still seeing the
     deprecation RuntimeWarning, `ddgs` isn't installed in this environment
     (`pip install ddgs`), and the deprecated package is known to be less
     reliable at returning results.
  1. A CONTROL query ("World Health Organization") sent directly to DDGS,
     bypassing all of this project's own code. If this returns 0 results,
     the problem is DDG access itself (network/rate-limiting/blocking) --
     nothing downstream can fix that by changing query construction.
  2. For each sampled claim: the raw fields pulled from the dataset (claim
     text, speaker, source_url) -- confirms we're actually reading what we
     think we're reading.
  3. If a source_url exists: fetch_source_article() run directly, with its
     (title, text length) shown -- this is the most failure-prone step in
     the real pipeline, so seeing it fail here in isolation is useful.
  4. The EXACT search query generate_search_query() produces -- visually
     inspect this for garbled/truncated text before trusting the search
     results it produces.
  5. That exact query sent directly to DDGS (bypassing search_claim_evidence
     entirely) -- raw hit count + first few URLs, isolating "DDG returned
     nothing for this specific query" from "DDG returned results but every
     page fetch after that failed."
  6. The full search_claim_evidence() call, for comparison against step 5 --
     a gap between the two numbers means candidates were found but their
     pages couldn't be fetched/parsed/excluded, not that search failed.
  7. (unless --retrieval-only) graph construction on whatever evidence was
     collected -- node/edge counts for the unaugmented and augmented graphs.

Ends with a summary table across every claim tested.
"""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def load_claims(data_path: str, n: int, seed: int, only_with_url: bool, only_without_url: bool):
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    for item in raw:
        label = item.get("label")
        if label not in ("Supported", "Refuted"):
            continue
        source_url = item.get("cached_original_claim_url") or item.get("original_claim_url") or ""
        if only_with_url and not source_url:
            continue
        if only_without_url and source_url:
            continue
        items.append({
            "text": item.get("claim", ""),
            "speaker": item.get("speaker") or "",
            "source_url": source_url,
            "label": label,
        })

    random.Random(seed).shuffle(items)
    return items[:n]


def check_ddgs_package():
    section("STEP 0: Which DuckDuckGo package is actually installed?")
    try:
        import ddgs  # noqa: F401
        print("Using 'ddgs' (the current, maintained package). Good.")
        return
    except ImportError:
        pass
    try:
        import duckduckgo_search  # noqa: F401
        print("Using the DEPRECATED 'duckduckgo_search' package.")
        print("Run:  pip install ddgs")
        print("The old package still works in many environments, but the new")
        print("one is where active fixes/updates are going.")
    except ImportError:
        print("NEITHER 'ddgs' NOR 'duckduckgo_search' is installed. Nothing in")
        print("this pipeline can search the web until one of them is. Run:")
        print("  pip install ddgs")
        sys.exit(1)


def control_query_test():
    section("STEP 1: Control query (bypasses this project's code entirely)")
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    query = "World Health Organization"
    print(f"Query: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        print(f"Raw hits: {len(results)}")
        for r in results[:3]:
            print(f"  - {r.get('href')}")
        if not results:
            print("\n*** 0 results for an extremely common, uncontroversial query. ***")
            print("*** This points to DDG blocking/rate-limiting this network/IP, ***")
            print("*** not to anything in this project's query construction.     ***")
            return False
        return True
    except Exception as e:
        print(f"ERROR during control query: {type(e).__name__}: {e}")
        print("\n*** DDGS itself raised an exception on the simplest possible  ***")
        print("*** query. Check network access, proxy settings, and whether  ***")
        print("*** 'pip install -U ddgs' changes anything.                   ***")
        return False


def run_one_claim(claim, idx, total, max_results, retrieval_only):
    from pipeline.retrieval import (
        fetch_source_article, generate_search_query, search_claim_evidence,
    )
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    section(f"CLAIM {idx}/{total}")
    print(f"text:       {claim['text'][:100]}")
    print(f"speaker:    {claim['speaker'] or '(none)'}")
    print(f"source_url: {claim['source_url'] or '(none -- will use claim-text fallback)'}")
    print(f"label:      {claim['label']}")

    result_row = {
        "text": claim["text"][:60],
        "has_url": bool(claim["source_url"]),
        "source_fetch_ok": None,
        "query": None,
        "raw_ddgs_hits": None,
        "final_evidence_count": None,
        "unaug_nodes": None,
        "aug_nodes": None,
        "aug_edges": None,
    }

    # --- Step 3: fetch source article directly, if we have a URL ---
    title = ""
    if claim["source_url"]:
        subsection("Step 3: fetch_source_article()")
        try:
            text, title = fetch_source_article(claim["source_url"])
            result_row["source_fetch_ok"] = bool(text)
            if text:
                print(f"OK -- title={title!r}, text_len={len(text)}")
                print(f"  text preview: {text[:150]!r}")
            else:
                print("FAILED -- fetch or extraction returned nothing for this URL.")
        except Exception as e:
            result_row["source_fetch_ok"] = False
            print(f"EXCEPTION: {type(e).__name__}: {e}")
            title = ""

    # --- Step 4: the exact query that would be generated ---
    subsection("Step 4: generate_search_query()")
    query = generate_search_query(
        raw_text=claim["text"], title=title or "", speaker_override=claim["speaker"] or None
    )
    result_row["query"] = query
    print(f"query: {query!r}")
    if len(query) < 10:
        print("*** WARNING: suspiciously short query. ***")

    # --- Step 5: that query sent directly to DDGS ---
    subsection("Step 5: raw DDGS().text(query) -- bypasses search_claim_evidence")
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results * 4))
        result_row["raw_ddgs_hits"] = len(raw_results)
        print(f"raw hits: {len(raw_results)}")
        for r in raw_results[:3]:
            print(f"  - {r.get('href')}")
    except Exception as e:
        result_row["raw_ddgs_hits"] = -1
        print(f"EXCEPTION: {type(e).__name__}: {e}")

    time.sleep(1)  # be polite between a raw call and the wrapped call below

    # --- Step 6: the full search_claim_evidence() call ---
    subsection("Step 6: search_claim_evidence() (full wrapper)")
    try:
        docs = search_claim_evidence(
            claim["text"], speaker=claim["speaker"] or None, source_url=claim["source_url"],
            max_results=max_results, verbose=True,
        )
        result_row["final_evidence_count"] = len(docs)
        print(f"final evidence docs: {len(docs)}")
        for d in docs:
            print(f"  - {d['url']}  ({len(d['text'])} chars)")
    except Exception as e:
        result_row["final_evidence_count"] = -1
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        docs = []

    # --- Step 7: graph construction (optional) ---
    if not retrieval_only:
        subsection("Step 7: graph construction")
        try:
            from pipeline.graph_building import build_article_graph, add_evidence_nodes
            from pipeline.augmentation import augment_with_cross_source

            base = build_article_graph(claim["text"])
            unaug = add_evidence_nodes(base, docs)
            aug, scored = augment_with_cross_source(unaug, claim["text"], docs)

            result_row["unaug_nodes"] = unaug.number_of_nodes()
            result_row["aug_nodes"] = aug.number_of_nodes()
            result_row["aug_edges"] = aug.number_of_edges()
            print(f"unaugmented: {unaug.number_of_nodes()} nodes, {unaug.number_of_edges()} edges")
            print(f"augmented:   {aug.number_of_nodes()} nodes, {aug.number_of_edges()} edges")
            print(f"scored docs: {len(scored)}")
        except Exception as e:
            print(f"EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()

    return result_row


def print_summary(rows):
    section("SUMMARY")
    header = f"{'has_url':>8} {'src_ok':>7} {'raw_hits':>9} {'final_ev':>9}"
    if any(r.get("aug_nodes") is not None for r in rows):
        header += f" {'aug_nodes':>10} {'aug_edges':>10}"
    print(header)
    for r in rows:
        line = (f"{str(r['has_url']):>8} {str(r['source_fetch_ok']):>7} "
                f"{str(r['raw_ddgs_hits']):>9} {str(r['final_evidence_count']):>9}")
        if r.get("aug_nodes") is not None:
            line += f" {str(r['aug_nodes']):>10} {str(r['aug_edges']):>10}"
        print(line)

    n = len(rows)
    zero_raw = sum(1 for r in rows if r["raw_ddgs_hits"] == 0)
    zero_final = sum(1 for r in rows if r["final_evidence_count"] == 0)
    print(f"\n{zero_raw}/{n} claims: DDGS itself returned 0 raw hits for the generated query "
          f"(search-construction problem, not a fetch problem)")
    print(f"{zero_final}/{n} claims: ended up with 0 final evidence docs")
    gap = zero_final - zero_raw
    if gap > 0:
        print(f"{gap} claim(s) had raw hits but still ended with 0 evidence "
              f"-- that's a page-fetch/parse/exclude-filter problem, not a search problem.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/dev.json", help="Path to an AVeriTeC JSON file")
    parser.add_argument("--n", type=int, default=8, help="Number of claims to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--retrieval-only", action="store_true",
                         help="Skip graph construction (no ENCODER/NLI_MODEL loading -- much faster startup)")
    parser.add_argument("--only-with-url", action="store_true", help="Only sample claims that have a source_url")
    parser.add_argument("--only-without-url", action="store_true", help="Only sample claims with no source_url")
    parser.add_argument("--claim", default=None, help="Test one specific claim string instead of sampling the dataset")
    parser.add_argument("--url", default="", help="source_url to pair with --claim")
    parser.add_argument("--speaker", default="", help="speaker to pair with --claim")
    parser.add_argument("--skip-control", action="store_true", help="Skip the Step 1 control query")
    args = parser.parse_args()

    if not Path("pipeline").exists():
        print("ERROR: no 'pipeline/' directory here. Run this script from the project root "
              "(the same directory models5.ipynb expects -- the one containing pipeline/ and data/).")
        sys.exit(1)

    check_ddgs_package()

    if not args.skip_control:
        ok = control_query_test()
        if not ok:
            print("\nControl query failed -- continuing anyway so you can see whether individual "
                  "claims behave any differently, but treat results below with suspicion.")

    if args.claim:
        claims = [{"text": args.claim, "speaker": args.speaker, "source_url": args.url, "label": "?"}]
    else:
        claims = load_claims(args.data, args.n, args.seed, args.only_with_url, args.only_without_url)
        if not claims:
            print(f"No claims loaded from {args.data} matching the given filters.")
            sys.exit(1)

    rows = []
    for i, claim in enumerate(claims, 1):
        row = run_one_claim(claim, i, len(claims), args.max_results, args.retrieval_only)
        rows.append(row)

    print_summary(rows)


if __name__ == "__main__":
    main()
