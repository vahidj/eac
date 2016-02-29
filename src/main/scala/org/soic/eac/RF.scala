package org.soic.eac

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}


/**
  * Created by vjalali on 2/27/16.
  */
object RF {
  def makeLibSVMLine(line: String): String =
  {
    val fields = line.split(",")
    return fields(6).toString + " 1:" + fields(0).toString + " 2:" + fields(1).toString +
      " 3:" + fields(2).toString + " 4:" + fields(3).toString + " 5:" + fields(4).toString + " 6:" + fields(5).toString
  }

  def main(args: Array[String]) = {
    val sc: SparkContext = new SparkContext()
    val rawData = sc.textFile("/Users/vjalali/Documents/Acad/eac/datasets/careval/car.data")
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val schemaString = "buying maint doors persons lug_boot safety acceptability"
    val schema = StructType(schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, true)))
    val rowRDD = rawData.map(_.split(",")).map(p => Row(p(0), p(1), p(2), p(3), p(4), p(5), p(6)))
    val carDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    var indexer = new StringIndexer().setInputCol("buying").setOutputCol("buyingIndex").fit(carDataFrame)
    var indexed = indexer.transform(carDataFrame)
    indexer = new StringIndexer().setInputCol("maint").setOutputCol("maintIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("doors").setOutputCol("doorsIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("persons").setOutputCol("personsIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("lug_boot").setOutputCol("lug_bootIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("safety").setOutputCol("safetyIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("acceptability").setOutputCol("acceptabilityIndex").fit(indexed)
    indexed = indexer.transform(indexed)

    val transformed = indexed.map(x => new LabeledPoint(x.get(13).asInstanceOf[Double],
      new DenseVector(Array(x.get(7).asInstanceOf[Double], x.get(8).asInstanceOf[Double], x.get(9).asInstanceOf[Double],
        x.get(10).asInstanceOf[Double], x.get(11).asInstanceOf[Double], x.get(12).asInstanceOf[Double]))))

    transformed.foreach(x => println(x.label))

    val splits = transformed.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    //trainingData.foreach(x => println(x.get(0).asInstanceOf[String]))
    //trainingData.map(x => new LabeledPoint(x.get(7).asInstanceOf[Double], new DenseVector(Array(0.2))))
    //val traidningRdd = trainingData.javaRDD.map(row => new LabeledPoint(row.toString.spli
    val numClasses = 4
    val categoricalFeaturesInfo = Map[Int, Int]((0,4),(1,4),(2,4),(3,3),(4,3),(5,3))
    val numTrees = 1 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 1
    val maxBins = 32

    val model = RandomForest.trainClassifier(trainingData.toJavaRDD(),
      numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, 10)
    val labelAndPreds = testData.map{
      point => val prediction = model.predict(point.features)
        println(point.label + " " + prediction)
        (point.label, prediction)
    }

    val testErr = labelAndPreds.filter(r => r._1 != r._2).count() * 1.0/testData.count()

    println("Test Error = " + testErr)
    println("Learned classification forest model:\n" + model.toDebugString)
    /*val data = rawData.map{line =>
        val values = line.split(",")
        val featureVector = Vectors.dense(1)
        val label = 2
        LabeledPoint(label, featureVector)
    }*/
    //MLUtils.
    //val data = MLUtils.loadLibSVMFile(sc, "/Users/vjalali/Documents/Acad/eac/datasets/careval/car.data")
    //data.foreach(println(_))
  }
}
