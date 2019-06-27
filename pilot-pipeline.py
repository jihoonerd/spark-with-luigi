import pyspark
import sys
from pyspark.sql import SparkSession
from collections import defaultdict
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.clustering import GaussianMixture, KMeans
from pyspark.ml import Pipeline
import time

if __name__ == "__main__":
    data_path = sys.argv[1]
    export_path = sys.argv[2]
    num_partition = int(sys.argv[3])
    seed = int(sys.argv[4])
    k = int(sys.argv[5])
    target = sys.argv[6]
    alg = sys.argv[7]

parquet_directory = data_path.replace('.csv', '') + '_parquet/'

# Open Session
spark = pyspark.sql.SparkSession.builder.master("local").appName("test").getOrCreate()

df = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true") \
    .load(data_path)

# Convert data type to parquet format
df.write.mode("overwrite").parquet(parquet_directory)
df = spark.read.option("header", "true")\
    .option("inferSchema", "true")\
    .parquet(parquet_directory)
print("Using PARQUET format")
data_length = df.count()
print("Data Size: ", data_length)

# Repartitioning
df = df.repartition(num_partition)

start = time.time()

# DATA TYPE SUMMARY
data_types = defaultdict(list)
for entry in df.schema.fields:
    data_types[str(entry.dataType)].append(entry.name)

# CATEGORIC PIPELINE

category_features = [var for var in data_types["StringType"] if var != target]
cat_missing = {}
for var in category_features:
    cat_missing[var] = "unknown"  # Impute categoric features
df = df.fillna(cat_missing)

useful_category_features = []
for var in category_features:
    # Drop if distinct values in a category column is greater than 15% of sample number.
    if df.select(var).distinct().count() < 0.15 * data_length:
        useful_category_features.append(var)

indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c), handleInvalid='skip')
            for c in useful_category_features]

encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="{0}_encoded".format(indexer.getOutputCol()))
            for indexer in indexers]

cat_assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders], outputCol="category")
cat_pipeline = Pipeline(stages=indexers + encoders + [cat_assembler])
df = cat_pipeline.fit(df).transform(df)

# NUMERIC PIPELINE
numeric_features = data_types["DoubleType"] + data_types["IntegerType"]
if target in numeric_features:
    numeric_features.remove(target)

for c in data_types["IntegerType"]:
    df = df.withColumn(c, df[c].cast("double"))

imputer = Imputer(inputCols=numeric_features, outputCols=[num +"_imputed" for num in numeric_features])
numeric_imputed = VectorAssembler(inputCols=imputer.getOutputCols(), outputCol="imputed")
scalers = StandardScaler(inputCol="imputed", outputCol="scaled")
num_assembler = VectorAssembler(inputCols=["scaled"], outputCol="numeric")
num_pipeline = Pipeline(stages=[imputer, numeric_imputed, scalers] + [num_assembler])
df = num_pipeline.fit(df).transform(df)

# FEATURES ASSEMBLE!
features_processed = VectorAssembler(inputCols=["category", "numeric"], outputCol="features")
tot_pipeline = Pipeline(stages=[features_processed])
processed = tot_pipeline.fit(df).transform(df)

# MODELING
if alg == 'GMM':
    gmm = GaussianMixture().setK(k).setFeaturesCol("features").setSeed(seed)
    print("=====" * 8)
    print(gmm.explainParams())
    print("=====" * 8)
    model = gmm.fit(processed)
elif alg == 'KMeans':
    kmm = KMeans().setK(k).setFeaturesCol("features").setSeed(seed)
    print("=====" * 8)
    print(kmm.explainParams())
    print("=====" * 8)
    model = kmm.fit(processed)
else:
    raise ValueError("no alg")

prediction = model.transform(processed)
prediction.select(numeric_features + category_features + [target, 'prediction']).coalesce(1).write.mode('overwrite').csv(export_path, header=True)
print("Result file is successfully generated at: ", export_path)

end=time.time()
print("ELAPSED TIME: ", end - start)