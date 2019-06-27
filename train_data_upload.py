from pyspark.sql import SparkSession
import sys


def main(argv):
    data_path = argv[1]
    parquet_path = argv[2]

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


if __name__ == '__main__':
    sys.exit(main(sys.argv))
