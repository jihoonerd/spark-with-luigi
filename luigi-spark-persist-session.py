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


class GlobalSettings(luigi.Config):

    # PATH
    data_path = luigi.Parameter(default="./usl_data/marketing.csv")
    parquet_path = luigi.Parameter(default="./luigi/marketing_parquet/")
    tf_path = luigi.Parameter(default="./luigi/marketing_transformed/")
    result_path = luigi.Parameter(default="./luigi/marketing_result/")

    # MODEL
    algorithm = luigi.Parameter(default="GMM")
    seed = luigi.Parameter(default="3")
    k = luigi.Parameter(default="10")

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

    luigi.build([TrainModel(settings=settings)])
    end = time.time()
    print("Elapsed Time: ", end-start)
#======================================================================




#
# class PreprocessData(SparkSubmitTask):
#
#     parquet_path = luigi.Parameter()
#     transformed_path = luigi.Parameter()
#
#     def requires(self):
#         return TrainDataUpload()
#
#     def run(self):
#         spark = pyspark.sql.SparkSession.builder.master("local").appName("test").getOrCreate()
#         df = spark.read.option("header", "true") \
#             .option("inferSchema", "true") \
#             .parquet(PARQUET_PATH)
#         df.repartition(200)
#         data_length = df.count()
#
#         # DATA TYPE SUMMARY
#         data_types = defaultdict(list)
#         for entry in df.schema.fields:
#             data_types[str(entry.dataType)].append(entry.name)
#
#         # CATEGORY PIPELINE
#         category_features = [var for var in data_types["StringType"] if var != TARGET]
#         cat_missing = {}
#         for var in category_features:
#             cat_missing[var] = "unknown"  # Impute categoric features
#         df = df.fillna(cat_missing)
#         useful_category_features = []
#         for var in category_features:
#             # Drop if distinct values in a category column is greater than 15% of sample number.
#             if df.select(var).distinct().count() < 0.15 * data_length:
#                 useful_category_features.append(var)
#
#         indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c), handleInvalid='skip')
#                     for c in useful_category_features]
#
#         encoders = [
#             OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="{0}_encoded".format(indexer.getOutputCol()))
#             for indexer in indexers]
#
#         cat_assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders],
#                                         outputCol="category")
#         cat_pipeline = Pipeline(stages=indexers + encoders + [cat_assembler])
#         df = cat_pipeline.fit(df).transform(df)
#
#         # NUMERIC PIPELINE
#         numeric_features = data_types["DoubleType"] + data_types["IntegerType"]
#         if TARGET in numeric_features:
#             numeric_features.remove(TARGET)
#
#         for c in data_types["IntegerType"]:
#             df = df.withColumn(c, df[c].cast("double"))
#
#         imputer = Imputer(inputCols=numeric_features, outputCols=[num + "_imputed" for num in numeric_features])
#         numeric_imputed = VectorAssembler(inputCols=imputer.getOutputCols(), outputCol="imputed")
#         scalers = StandardScaler(inputCol="imputed", outputCol="scaled")
#         num_assembler = VectorAssembler(inputCols=["scaled"], outputCol="numeric")
#         num_pipeline = Pipeline(stages=[imputer, numeric_imputed, scalers] + [num_assembler])
#         df = num_pipeline.fit(df).transform(df)
#
#         # FEATURES ASSEMBLE!
#         features_processed = VectorAssembler(inputCols=["category", "numeric"], outputCol="features")
#         tot_pipeline = Pipeline(stages=[features_processed])
#         processed = tot_pipeline.fit(df).transform(df)
#         processed.write.mode("overwrite").parquet(TRANSFORMED_PATH)
#
#         feature_info = {"numeric_features": numeric_features, "category_features": category_features}
#         with open("feature_info.pickle", "wb") as handle:
#             pickle.dump(feature_info, handle)
#
#     def output(self):
#         return luigi.LocalTarget(self.transformed_path + "_SUCCESS")
#
#     def app_options(self):
#         # :func:`~luigi.task.Task.input` returns the targets produced by the tasks in
#         # `~luigi.task.Task.requires`.
#         return [self.parquet_path, self.transformed_path]
#
#
# class Train(PySparkTask):
#
#     seed = luigi.IntParameter()
#     result_path = luigi.Parameter()
#
#     def requires(self):
#         return PreprocessData()
#
#     def run(self):
#         spark = pyspark.sql.SparkSession.builder.master("local").appName("test").getOrCreate()
#         processed = spark.read.option("header", "true") \
#             .option("inferSchema", "true") \
#             .parquet(TRANSFORMED_PATH)
#         processed.repartition(200)
#
#
#         # MODELING
#         if ALGORITHM == 'GMM':
#             gmm = GaussianMixture().setK(K).setFeaturesCol("features").setSeed(self.seed)
#             print("=====" * 8)
#             print(gmm.explainParams())
#             print("=====" * 8)
#             model = gmm.fit(processed)
#         elif ALGORITHM == 'KMeans':
#             kmm = KMeans().setK(K).setFeaturesCol("features").setSeed(self.seed)
#             print("=====" * 8)
#             print(kmm.explainParams())
#             print("=====" * 8)
#             model = kmm.fit(processed)
#         else:
#             raise ValueError("no alg")
#
#         prediction = model.transform(processed)
#         with open("feature_info.pickle", "rb") as handle:
#             feature_info = pickle.load(handle)
#
#         numeric_features = feature_info["numeric_features"]
#         category_features = feature_info["category_features"]
#         prediction.select(numeric_features + category_features + [TARGET, 'prediction']).coalesce(1).write.mode(
#             'overwrite').csv(self.result_path, header=True)
#         print("Result file is successfully generated at: ", self.result_path)
#     def output(self):
#         return luigi.LocalTarget(RESULT_PATH)


