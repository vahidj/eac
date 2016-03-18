package org.soic.eac

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{TrainValidationSplit, ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
/**
  * Created by vjalali on 3/17/16.
  */
class BalanceReader extends reader{
  def Output(indexed: DataFrame): DataFrame= {
    val transformedDf = indexed.drop("class")
      var assembler = new VectorAssembler().setInputCols(Array("left-weight", "left-distance", "right-weight", "right-distance"))
      .setOutputCol("features")
      var output = assembler.transform(transformedDf)
      return output
  }

  def DFTransformed(indexed: DataFrame): RDD[LabeledPoint] = {
    val transformed = indexed.map(x => new LabeledPoint(x.get(0).asInstanceOf[Double],
      new DenseVector(Array(x.get(1).asInstanceOf[Double], x.get(2).asInstanceOf[Double], x.get(3).asInstanceOf[Double],
        x.get(4).asInstanceOf[Double]))))
    return transformed
  }
  def Indexed(FilePath:String, schemaString:String, sc: SparkContext): DataFrame= {
    val rawData = sc.textFile(FilePath)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val schema = StructType(schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, true)))
    val rowRDD = rawData.map(_.split(",")).map(p => Row(p(0), p(1), p(2), p(3), p(4)))
    val balanceDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    var indexer = new StringIndexer().setInputCol("class").setOutputCol("classIndex").fit(balanceDataFrame)
    var indexed = indexer.transform(balanceDataFrame)
    return indexed
  }

  override def numberOfClasses: Int = 3

  override def categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]((0,5),(1,5),(2,5),(3,5))
}
