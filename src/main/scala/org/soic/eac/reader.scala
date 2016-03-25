package org.soic.eac
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext

trait Reader {
  def Output(indexed: DataFrame):DataFrame
  def Indexed(FilePath:String, schemaString:String, sc: SparkContext): DataFrame
  def DFTransformed(indexed: DataFrame): RDD[LabeledPoint]
  def numberOfClasses: Int
  def categoricalFeaturesInfo: Map[Int, Int]
  def numericalFeaturesInfo: Map[Int, Double]
  def dataSchema: String
  }
