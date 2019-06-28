from pyspark.sql import SparkSession
from collections import defaultdict
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
import pickle
import sys


def main(argv):

    parquet_path = argv[1]
    tf_path = argv[2]
    target = argv[3]

    spark = SparkSession.builder.config("spark.driver.memory", "32g").config("spark.executor.memory", "32g")\
        .config("spark.driver.maxResultSize", "20g").getOrCreate()

    with open("transform_spark.txt", "w") as file:
        file.write("spark context" + str(spark.sparkContext))
        file.write("===SeessionID===")
        file.write(str(id(spark)))

    print(spark)
    df = spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .parquet(parquet_path)
    df.repartition(10)

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

    with open("./feature_info.pickle", "wb") as handle:
        pickle.dump(feature_info, handle)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

