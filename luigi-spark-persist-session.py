import luigi
from pyspark.sql import SparkSession
import os
from luigi.contrib.spark import SparkSubmitTask
from pyspark.sql import SparkSession
from collections import defaultdict
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import GaussianMixture, KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
import pickle
import sys
import pandas as pd


class GlobalSettings(luigi.Config):

    # PATH
    data_path = luigi.Parameter(default="./usl_data/marketing.csv")
    parquet_path = luigi.Parameter(default="./luigi/marketing_parquet/")
    tf_path = luigi.Parameter(default="./luigi/marketing_transformed/")
    result_path = luigi.Parameter(default="./luigi/marketing_result/")

    # MODEL
    algorithm = luigi.Parameter(default="GMM")
    seed = luigi.Parameter(default="3")
    k = luigi.Parameter(default="6")

    # Optional
    target = luigi.Parameter(default="insurance_subscribe")



class TrainDataUpload(luigi.Task):

    settings = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.settings.parquet_path + "_SUCCESS")

    def run(self):

        data_path = self.settings.data_path
        parquet_path = self.settings.parquet_path

        spark = SparkSession.builder.getOrCreate()

        with open("upload_spark.txt", "w") as file:
            file.write("spark context" + str(spark.sparkContext))
            file.write("===SeessionID===")
            file.write(str(id(spark)))

        df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(data_path)

        print("Data Size: ", df.count())
        df.write.mode("overwrite").parquet(parquet_path)
        print("Save parquet format complete.")


class Transform(luigi.Task):

    settings = luigi.Parameter()

    def requires(self):
        return TrainDataUpload(settings=settings)

    def output(self):
        return luigi.LocalTarget(self.settings.tf_path + "_SUCCESS")

    def run(self):
        parquet_path = self.settings.parquet_path
        tf_path = self.settings.tf_path
        target = self.settings.target

        spark = SparkSession.builder.getOrCreate()

        with open("transform_spark.txt", "w") as file:
            file.write("spark context" + str(spark.sparkContext))
            file.write("===SeessionID===")
            file.write(str(id(spark)))

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

        with open("./feature_info.pickle", "wb") as handle:
            pickle.dump(feature_info, handle)


class TrainModel(luigi.Task):

    settings = luigi.Parameter()

    def requires(self):
        return [Transform(settings=settings)]

    def output(self):
        return luigi.LocalTarget(self.settings.result_path + "_SUCCESS")

    def run(self):

        tf_path = self.settings.tf_path
        algorithm =  self.settings.algorithm
        seed = int(self.settings.seed)
        k = int(self.settings.k)
        result_path = self.settings.result_path
        target = self.settings.target

        spark = SparkSession.builder.getOrCreate()


        with open("train_spark.txt", "w") as file:
            file.write("spark context" + str(spark.sparkContext))
            file.write("===SeessionID===")
            file.write(str(id(spark)))

        df = spark.read.option("header", "true") \
            .option("inferSchema", "true") \
            .parquet(tf_path)
        df.repartition(200)

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
    # luigi.run()
    import time
    start = time.time()
    settings = GlobalSettings()
    spark = SparkSession.builder.config("spark.driver.memory", "32g").config("spark.executor.memory", "32g").getOrCreate()

    with open("main_spark.txt", "w") as file:
        file.write("spark context" + str(spark.sparkContext))
        file.write("===SeessionID===")
        file.write(str(id(spark)))

    luigi.build([TrainModel(settings=settings)], local_scheduler=True)
    end = time.time()
    elapsed = end - start
    print("Elapsed Time: ", elapsed)
    logtable = pd.read_csv("./timelog.csv")
    logtable.append(pd.DataFrame([[GlobalSettings().data_path, "multiple session", elapsed,
                                   GlobalSettings().algorithm, GlobalSettings().k, GlobalSettings().seed]],
                                 columns=logtable.columns),
                    ignore_index=True).to_csv("./timelog.csv", index=False)

