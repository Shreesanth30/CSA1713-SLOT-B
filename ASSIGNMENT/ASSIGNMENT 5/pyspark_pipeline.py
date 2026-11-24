"""Task A4 / B3.3 — the RDD pipeline in real PySpark.  NOT executed in the authoring sandbox (PySpark could not be installed offline);
run on your own machine:   pip install pyspark   then   python pyspark_pipeline.py 1000000 4
It implements the same pipeline stages as the one measured with the mini-RDD engine (rdd_mini.py): same parse, clean, reduceByKey, takeOrdered, anomaly count."""
import sys, time, random
from pyspark import SparkContext, StorageLevel

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10**6
P = int(sys.argv[2]) if len(sys.argv) > 2 else 4
sc = SparkContext("local[*]", "RetrievalLogPipeline")

def gen(i):                                    # synthetic 'docId,timestamp,similarity' lines (seeded) - replace with sc.textFile(path, P)
    rnd = random.Random(i); d = int(rnd.random() ** 4 * 5000); v = rnd.gauss(0, 8 if d % 500 == 0 else 1) + d % 50
    return f"{d},{i},{-999 if rnd.random() < 0.01 else v:.3f}"
raw = sc.parallelize(range(N), P).map(gen)                                     # narrow  (lazy)  == sc.textFile(path, P)
parsed = raw.map(lambda l: l.split(",")).map(lambda a: (int(a[0]), int(a[1]), float(a[2])))   # narrow
clean = (parsed.filter(lambda r: r[2] > -900)                                  # narrow : drop invalid readings
               .map(lambda r: (r[0], r[2] * (1 + 0.01 * (r[0] % 7)) + (r[0] % 3))))                # narrow : calibrate -> (doc, value)
clean.persist(StorageLevel.MEMORY_AND_DISK)                                    # cached: reused by TWO actions below
stats = (clean.map(lambda kv: (kv[0], (kv[1], kv[1], kv[1] * kv[1], 1)))                      # narrow
              .reduceByKey(lambda a, b: (a[0] + b[0], max(a[1], b[1]), a[2] + b[2], a[3] + b[3])))   # WIDE: the only shuffle (map-side combine)
per_doc = stats.mapValues(lambda s: (s[0] / s[3], s[1], s[2] / s[3] - (s[0] / s[3]) ** 2))             # narrow -> (mean, max, variance)
t0 = time.time()
table = per_doc.collect()                                                      # ACTION 1: triggers stage 0 (parse..combine) + stage 1 (reduce)
top10 = per_doc.takeOrdered(10, key=lambda kv: -kv[1][2])                      # ACTION 2: ten noisiest docs (partial top-10 per partition, no shuffle)
anomalies = clean.filter(lambda kv: kv[1] > 40.0).count()                      # ACTION 3: re-uses the cached `clean` RDD
print(f"N={N} P={P} docs={len(table)} anomalies={anomalies} wall={time.time()-t0:.2f}s")
print("top-10 noisiest docs:", [k for k, _ in top10])
print(per_doc.toDebugString().decode())                                       # lineage graph (shows the ShuffledRDD boundary)
sc.stop()
