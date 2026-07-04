"""The ONE cross-node operation: group-by-k-mer SUM. Two interchangeable backends
(proven to give identical results — see tmp/dist_test):

  reduce_pandas : single-process. Use for debugging, tests, and Regime-1 when the
                  partials fit one node. No framework.
  reduce_spark  : cross-node distributed (Spark). Use at scale; needs the kmx[spark]
                  extra. Reads per-node parquet partials, groupBy(key).sum().

Both take a list of per-node partial tables (key columns + 'count') and return the
global summed table. Swap backends without changing any other stage.
"""
import pandas as pd


def reduce_pandas(partials, key):
    """Sum 'count' per k-mer across all partial tables (local, single process)."""
    return pd.concat(list(partials), ignore_index=True).groupby(
        key, as_index=False)["count"].sum()


def reduce_spark(parquet_paths, key, *, master, driver_host, local_dir):
    """Cross-node distributed sum. Lazily imports pyspark (kmx[spark] extra).
    Returns a pandas DataFrame of the global (key, count)."""
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    sp = (SparkSession.builder.master(master).appName("kmx-reduce")
          .config("spark.ui.enabled", "false").config("spark.driver.host", driver_host)
          .config("spark.local.dir", local_dir).getOrCreate())
    try:
        df = sp.read.parquet(*list(parquet_paths))
        out = df.groupBy(*key).agg(F.sum("count").alias("count")).toPandas()
    finally:
        sp.stop()
    return out
