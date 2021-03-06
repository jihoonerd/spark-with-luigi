import luigi
from pyspark.sql import SparkSession
import os
from luigi.contrib.spark import SparkSubmitTask
import pandas as pd

class GlobalSettings(luigi.Config):

    # PATH
    data_path = luigi.Parameter(default="./usl_data/mnist_test.csv")
    parquet_path = luigi.Parameter(default="./luigi/mnist_test_parquet/")
    tf_path = luigi.Parameter(default="./luigi/mnist_test_transformed/")
    result_path = luigi.Parameter(default="./luigi/mnist_test_result/")

    # MODEL
    algorithm = luigi.Parameter(default="GMM")
    seed = luigi.Parameter(default="3")
    k = luigi.Parameter(default="10")

    # Optional
    target = luigi.Parameter(default="label")



class TrainDataUpload(SparkSubmitTask):

    app = 'train_data_upload.py'
    master = 'local[*]'

    def output(self):
        return luigi.LocalTarget(GlobalSettings().parquet_path + "_SUCCESS")

    def app_options(self):
        return [GlobalSettings().data_path, GlobalSettings().parquet_path]


class Transform(SparkSubmitTask):

    app = 'transform_features.py'
    master = 'local[*]'

    def requires(self):
        return TrainDataUpload()

    def output(self):
        return luigi.LocalTarget(GlobalSettings().tf_path + "_SUCCESS")

    def app_options(self):
        return [GlobalSettings().parquet_path, GlobalSettings().tf_path, GlobalSettings().target]


class TrainModel(SparkSubmitTask):

    app = 'train_model.py'
    master = 'local[*]'

    def requires(self):
        return [Transform()]

    def output(self):
        return luigi.LocalTarget(GlobalSettings().result_path + "_SUCCESS")

    def app_options(self):
        return [GlobalSettings().tf_path,
                GlobalSettings().algorithm,
                GlobalSettings().seed,
                GlobalSettings().k,
                GlobalSettings().result_path,
                GlobalSettings().target]


if __name__ == '__main__':
    # luigi.run()
    import time
    start = time.time()
    # luigi.run()
    luigi.build([TrainModel()], local_scheduler=True)
    end = time.time()
    elapsed = end-start
    print("Elapsed Time: ", elapsed)
    logtable = pd.read_csv("./timelog.csv")
    logtable.append(pd.DataFrame([[GlobalSettings().data_path, "multiple session", elapsed,
                               GlobalSettings().algorithm, GlobalSettings().k, GlobalSettings().seed]], columns=logtable.columns),
                    ignore_index=True).to_csv("./timelog.csv", index=False)


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


