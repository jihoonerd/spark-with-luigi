from pyspark.sql import SparkSession
from collections import defaultdict
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.clustering import GaussianMixture, KMeans
from pyspark.ml import Pipeline
import time

# PATH
data_path = "./usl_data/marketing.csv"
parquet_path = "./luigi/marketing_parquet/"
tf_path = "./luigi/marketing_transformed/"
result_path = "./luigi/marketing_result/"

# MODEL
algorithm = "GMM"
seed = int("3")
k = int("6")

# Optional
target = "insurance_subscribe"

start = time.time()
spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "32g").config("spark.executor.memory", "32g").getOrCreate()


spark = SparkSession.builder.getOrCreate()

df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(data_path)

print("Data Size: ", df.count())
df.write.mode("overwrite").parquet(parquet_path)
print("Save parquet format complete.")


df = spark.read.option("header", "true") \
            .option("inferSchema", "true") \
            .parquet(parquet_path)
df.repartition(200)

# DATA TYPE SUMMARY
data_types = defaultdict(list)
for entry in df.schema.fields:
    data_types[str(entry.dataType)].append(entry.name)

# NUMERIC PIPELINE
numeric_features = data_types["DoubleType"] + data_types["IntegerType"]
if target in numeric_features:
    numeric_features.remove(target)

for c in data_types["IntegerType"]:
    df = df.withColumn(c, df[c].cast("double"))

imputer = Imputer(inputCols=numeric_features, outputCols=[num + "_imputed" for num in numeric_features])
numeric_imputed = VectorAssembler(inputCols=imputer.getOutputCols(), outputCol="imputed")
scaler = StandardScaler(inputCol="imputed", outputCol="scaled")
num_assembler = VectorAssembler(inputCols=["scaled"], outputCol="numeric")
num_pipeline = Pipeline(stages=[imputer, numeric_imputed, scaler] + [num_assembler])
df = num_pipeline.fit(df).transform(df)

# CATEGORY PIPELINE
category_features = [var for var in data_types["StringType"] if var != target]
cat_missing = {}
for var in category_features:
    cat_missing[var] = "unknown"  # Impute category features
df = df.fillna(cat_missing)
useful_category_features = []

for var in category_features:
    # Drop if distinct values in a category column is greater than 15% of sample number.
    if df.select(var).distinct().count() < 0.15 * df.count():
        useful_category_features.append(var)

indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c), handleInvalid='skip')
            for c in useful_category_features]

encoders = [
    OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="{0}_encoded".format(indexer.getOutputCol()))
    for indexer in indexers]

cat_assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders],
                                outputCol="category")
cat_pipeline = Pipeline(stages=indexers + encoders + [cat_assembler])
df = cat_pipeline.fit(df).transform(df)

# Integrate features
features_processed = VectorAssembler(inputCols=["category", "numeric"], outputCol="features")
tot_pipeline = Pipeline(stages=[features_processed])
processed = tot_pipeline.fit(df).transform(df)
processed.write.mode("overwrite").parquet(tf_path)

feature_info = {"numeric_features": numeric_features, "category_features": category_features}


# MODELING
if algorithm == 'GMM':
    gmm = GaussianMixture().setK(k).setFeaturesCol("features").setSeed(seed)
    print("=====" * 8)
    print(gmm.explainParams())
    print("=====" * 8)
    model = gmm.fit(processed)
elif algorithm == 'KMeans':
    kmm = KMeans().setK(k).setFeaturesCol("features").setSeed(seed)
    print("=====" * 8)
    print(kmm.explainParams())
    print("=====" * 8)
    model = kmm.fit(processed)
else:
    raise ValueError("no alg")

prediction = model.transform(processed)


prediction.select(feature_info["numeric_features"] + feature_info["category_features"] +
                  [target, 'prediction']).coalesce(1).write.mode(
    'overwrite').csv(result_path, header=True)
print("Result file is successfully generated at: ", result_path)

end=time.time()
print("ELAPSED TIME: ", end - start)