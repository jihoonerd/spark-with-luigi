from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import GaussianMixture, KMeans

from pyspark.ml import Pipeline
import pickle
import sys


def main(argv):

    tf_path = argv[1]
    algorithm = argv[2]
    seed = int(argv[3])
    k = int(argv[4])
    result_path = argv[5]
    target = argv[6]

    spark = SparkSession.builder.config("spark.driver.memory", "32g").config("spark.executor.memory", "32g")\
        .config("spark.driver.maxResultSize", "20g").getOrCreate()
    with open("train_spark.txt", "w") as file:
        file.write("spark context" + str(spark.sparkContext))
        file.write("===SeessionID===")
        file.write(str(id(spark)))

    df = spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .parquet(tf_path)
    df.repartition(10)

    # MODELING
    if algorithm == 'GMM':
        gmm = GaussianMixture().setK(k).setFeaturesCol("features").setSeed(seed)
        print("=====" * 8)
        print(gmm.explainParams())
        print("=====" * 8)
        model = gmm.fit(df)
    elif algorithm == 'KMeans':
        kmm = KMeans().setK(k).setFeaturesCol("features").setSeed(seed)
        print("=====" * 8)
        print(kmm.explainParams())
        print("=====" * 8)
        model = kmm.fit(df)
    else:
        raise ValueError("no alg")

    prediction = model.transform(df)

    with open("./feature_info.pickle", "rb") as handle:
        features_info = pickle.load(handle)

    prediction.select(features_info["numeric_features"] + features_info["category_features"] +
                      [target, 'prediction']).coalesce(1).write.mode(
        'overwrite').csv(result_path, header=True)
    print("Result file is successfully generated at: ", result_path)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
