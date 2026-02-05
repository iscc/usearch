import argparse
import time

from usearch.index import Index


def _view_once(path: str, enable_key_lookups: bool) -> Index:
    meta = Index.metadata(path)
    if meta is None:
        raise SystemExit(f"Index.metadata() returned None for: {path}")

    idx = Index(
        ndim=meta["dimensions"],
        metric=meta["kind_metric"],
        dtype=meta["kind_scalar"],
        enable_key_lookups=enable_key_lookups,
    )

    t0 = time.perf_counter()
    idx.view(path)
    t1 = time.perf_counter()

    print(f"view(enable_key_lookups={enable_key_lookups})  {t1 - t0:.6f}s")

    if enable_key_lookups:
        # First key access is expected to trigger any lazy key indexing.
        t0 = time.perf_counter()
        _ = idx.contains(0)
        t1 = time.perf_counter()
        print(f"contains(0) first-call                {t1 - t0:.6f}s")

        t0 = time.perf_counter()
        _ = idx.contains(0)
        t1 = time.perf_counter()
        print(f"contains(0) second-call               {t1 - t0:.6f}s")

    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to a .usearch index shard")
    args = ap.parse_args()

    _view_once(args.path, enable_key_lookups=True)
    _view_once(args.path, enable_key_lookups=False)


if __name__ == "__main__":
    main()

