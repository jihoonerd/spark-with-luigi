from pyspark.sql import SparkSession
from collections import defaultdict
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
import pickle
import sys


def main(argv):

    parquet_path = argv[1]
    category_tf_path = argv[2]
    target = argv[3]

    spark = SparkSession.builder.getOrCreate()
    file=open("./tran")
    df = spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .parquet(parquet_path)
    df.repartition(200)

    # DATA TYPE SUMMARY
    data_types = defaultdict(list)
    for entry in df.schema.fields:
        data_types[str(entry.dataType)].append(entry.name)

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

    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="{0}_encoded".format(indexer.getOutputCol()))
                for indexer in indexers]

    cat_assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders],
                                    outputCol="category")
    cat_pipeline = Pipeline(stages=indexers + encoders + [cat_assembler])
    df = cat_pipeline.fit(df).transform(df)
    df.select(["category"]).write.mode("overwrite").parquet(category_tf_path)
    category_feature_info = {"category_features": category_features}
    with open(category_tf_path + "category_feature_info.pickle", "wb") as handle:
        pickle.dump(category_feature_info, handle)


if __name__ == '__main__':
    sys.exit(main(sys.argv))

