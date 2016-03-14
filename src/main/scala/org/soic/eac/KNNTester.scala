package org.soic.eac

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
import org.soic.eac.EACConfig._

/**
  * Created by vjalali on 2/27/16.
  */
object KNNTester {
  System.setProperty("hadoop.home.dir", "c:/winutil")
  def makeLibSVMLine(line: String): String =
  {
    val fields = line.split(",")
    return fields(6).toString + " 1:" + fields(0).toString + " 2:" + fields(1).toString +
      " 3:" + fields(2).toString + " 4:" + fields(3).toString + " 5:" + fields(4).toString + " 6:" + fields(5).toString
  }

  def main(args: Array[String]) = {
    val sc: SparkContext = new SparkContext()
    val filePathAdult="/Users/vjalali/Documents/Acad/eac/datasets/adult/adult.data"
    println(EACConfig.BASEPATH)
    val filePathCar= EACConfig.BASEPATH + "datasets/careval/car.data"
    val schemaStringAdult = "age workclass fnlwgt education education-num marital occupation relationship race sex capital-gain capital-loss hours-per-week country income"
    val schemaStringCar= "buying maint doors persons lug_boot safety acceptability"
    val readr= new carReader // new adultReader
    val indexed = readr.Indexed(filePathCar, schemaStringCar,sc)
    val transformed = readr.DFTransformed(indexed)
    val output = readr.Output(indexed)
    
    val splits = transformed.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    val numClasses = 4
    val categoricalFeaturesInfo = Map[Int, Int]((0,4),(1,4),(2,4),(3,3),(4,3),(5,3))
    val numTrees = 100 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val nfolds: Int = 20
    val knn = new EAC(1, 10, trainingData, testData, transformed)
    //knn.persistNearestNeighbors()

    //val paramGrid = new ParamGridBuilder().addGrid(knn.k, Array(1,2,3,4,5,6,7)).build()
    //val paramGrid = new ParamGridBuilder().addGrid(rf.numTrees, Array(1,5,10,30,60,90)).addGrid(rf.maxDepth, Array(1,2,3,4,5,6,7,8,9,10))
    //  .addGrid(rf.maxBins, Array(30, 60, 90)).build()
    /*val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(rf)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)*/

    //val cv = new CrossValidator().setEstimator(knn).setEvaluator(new MulticlassClassificationEvaluator())
    //  .setEstimatorParamMaps(paramGrid)
    //  .setNumFolds(nfolds)

    val model = RandomForest.trainClassifier(trainingData.toJavaRDD(),
      numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, 10)

    //println("++++++++++++++++++++++++++++++++++++++++\n"+cv.fit(output).bestModel.params.toString())
    //cv.fit(output).bestModel.params.foreach(x => println(x))
    
    
    // Extracting best model params
    
    //val cvModel= cv.fit(output)
    //val paramMap = {cvModel.getEstimatorParamMaps
    //       .zip(cvModel.avgMetrics)
    //       .maxBy(_._2)
    //       ._1}
    //println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    //println("Best Model params\n"+ paramMap)

    //paramMap.toSeq.filter(_.param.name == "maxBins")(0).value

    knn.train()

    /*println(knn.predict(testData.first().features))
    val labelAndPreds = testData.map{
      point => val prediction = knn.predict(point.features)
        //println(point.label + " " + prediction)
        (point.label, prediction)
    }*/
    val labeleAndPredsRF = testData.map{
      point => val prediction = model.predict(point.features)
        (point.label, prediction)
    }
    val labelAndPreds = knn.getPredAndLabels()
    //println(labelAndPreds)
    //println(labelAndPreds.filter(r => r._1 != r._2).count())
    val testErr = labelAndPreds.filter(r => r._1 != r._2).length * 1.0/testData.count()
    val testErrRF = labeleAndPredsRF.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0/testData.count()
    println("KNN Test Error = " + testErr + " RF test error = " + testErrRF)
    //println("Learned classification forest model:\n" + model.toDebugString)
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
