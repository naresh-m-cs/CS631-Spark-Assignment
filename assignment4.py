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


spark.stop()
