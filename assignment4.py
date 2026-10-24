import sys, re, json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def _fmt(v):
    """Convert one cell value to the text that is printed."""
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_df(task_name, df, sort_cols):
    """Print a deterministically sorted DataFrame for autograding."""
    ordered = df.orderBy(*sort_cols)
    print(f"BEGIN_{task_name}")
    print("\t".join(ordered.columns))
    for row in ordered.toLocalIterator():
        print("\t".join(_fmt(v) for v in row))
    print(f"END_{task_name}")
    print()  # empty line between blocks, for readability


PLACES = ["delhi", "mumbai", "kolkata", "chennai", "bengaluru",
          "hyderabad", "lucknow", "srinagar", "kashmir", "punjab",
          "gujarat", "maharashtra", "kerala", "karnataka", "bengal",
          "assam", "bihar", "rajasthan", "pakistan", "china",
          "noida", "mathura"]

spark = SparkSession.builder.appName("GeoCoverage").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')

data_path = sys.argv[1]

articles = (sc.wholeTextFiles(data_path.rstrip("/") + "/*.json")
              .map(lambda kv: json.loads(kv[1]))
              .map(lambda d: (str(d["article_id"]), d.get("article_body") or "")))
articles.cache()

# Task 1
def places_in(rec):
    aid, body = rec
    words = set(re.findall(r'[a-z]+', body.lower()))
    return [(p, aid) for p in PLACES if p in words]

pairs_rdd = articles.flatMap(places_in).distinct()
pairs_rdd.cache()

total_pairs = pairs_rdd.count()
first_rows = pairs_rdd.sortBy(lambda x: (x[0], x[1])).take(20)
print("BEGIN_TASK1_RDD")
print(total_pairs)
for p, a in first_rows:
    print(f"{p}\t{a}")
print("END_TASK1_RDD")
print()

# Task 2
pairs_df = pairs_rdd.toDF(["place", "article_id"])
pairs_df.cache()
print_df('TASK2_PAIRS',
         pairs_df.orderBy('place', 'article_id').limit(20),
         [F.col('place').asc(), F.col('article_id').asc()])

# Task 3
art_df = articles.toDF(["article_id", "article_body"])

tf_df = (art_df
         .select("article_id",
                 F.explode(F.split(F.lower(F.col("article_body")), '[^a-z]+')).alias("place"))
         .filter(F.col("place").isin(PLACES))
         .groupBy("article_id", "place")
         .agg(F.count("*").alias("term_frequency")))
tf_df.cache()

print_df('TASK3_FREQUENCIES', tf_df,
         [F.col('article_id').asc(), F.col('place').asc()])

w = Window.partitionBy("article_id")
dominant_df = (tf_df
               .withColumn("max_tf", F.max("term_frequency").over(w))
               .filter(F.col("term_frequency") == F.col("max_tf"))
               .select("article_id", "place", "term_frequency"))

print_df('TASK3_DOMINANT', dominant_df,
         [F.col('article_id').asc(), F.col('place').asc()])

# Task 4
blind_df = (pairs_df
            .groupBy("place")
            .agg(F.count("*").alias("n"), F.min("article_id").alias("article_id"))
            .filter(F.col("n") == 1)
            .select("place", "article_id"))

print_df('TASK4_BLIND_SPOTS', blind_df,
         [F.col('place').asc(), F.col('article_id').asc()])

# Task 5
place_counts = (pairs_df.groupBy("place").agg(F.count("*").alias("article_count")))

#perform the self join operation
cooccurrences = (
    pairs_df.alias("a")
    .join(
        pairs_df.alias("b"),
        (F.col("a.article_id") == F.col("b.article_id")) &
        (F.col("a.place") < F.col("b.place"))
    )
    .select(
        F.col("a.place").alias("p1"),
        F.col("b.place").alias("p2")
    )
    .groupBy("p1", "p2")
    .agg(F.count("*").alias("cooccurrence_count"))
)

counts1 = place_counts.select(
    F.col("place").alias("p1"),
    F.col("article_count").alias("n1")
)

counts2 = place_counts.select(
    F.col("place").alias("p2"),
    F.col("article_count").alias("n2")
)

jaccard_df = (
    cooccurrences
    .join(counts1, "p1")
    .join(counts2, "p2")
    .select(
        "p1",
        "p2",
        (
            F.col("cooccurrence_count").cast("double") /
            (F.col("n1") + F.col("n2") - F.col("cooccurrence_count"))
        ).alias("jaccard"),
        "cooccurrence_count"
    )
).cache()

#printing the results: 
top_order = [
    F.col("jaccard").desc(),
    F.col("p1").asc(),
    F.col("p2").asc()
]

bottom_order = [
    F.col("jaccard").asc(),
    F.col("p1").asc(),
    F.col("p2").asc()
]

print_df(
    "TASK5_TOP5",
    jaccard_df.orderBy(*top_order).limit(5),
    top_order
)

print_df(
    "TASK5_BOTTOM5",
    jaccard_df.orderBy(*bottom_order).limit(5),
    bottom_order
)

# Task 6
coverage_df = (
    tf_df
    .groupBy("place")
    .agg(F.sum("term_frequency").alias("total_mentions"))
    .join(place_counts, "place")
    .select("place", "total_mentions", "article_count")
)

coverage_order = [
    F.col("total_mentions").desc(),
    F.col("place").asc()
]

print_df(
    "TASK6_TOP5",
    coverage_df.orderBy(*coverage_order).limit(5),
    coverage_order
)

# Task 7
distinct_counts = (
    pairs_df
    .groupBy("article_id")
    .agg(F.count("*").alias("distinct_place_count"))
)

crowded_df = (
    art_df.select("article_id").distinct()
    .join(distinct_counts, "article_id", "left")
    .fillna({"distinct_place_count": 0})
    .select("article_id", "distinct_place_count")
)

crowded_order = [
    F.col("distinct_place_count").desc(),
    F.col("article_id").asc()
]

print_df(
    "TASK7_TOP10",
    crowded_df.orderBy(*crowded_order).limit(10),
    crowded_order
)

spark.stop()
