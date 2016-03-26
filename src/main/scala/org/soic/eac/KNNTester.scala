package org.soic.eac

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{TrainValidationSplit, ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.classification.{SVMWithSGD, NaiveBayes, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.soic.eac.EACConfig._
import java.io._

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
    //val filePathAdult="/Users/vjalali/Documents/Acad/eac/datasets/adult/adult.data"
    println(EACConfig.BASEPATH)
    val filePathCar= EACConfig.BASEPATH + "datasets/careval/car.data"
    val filePathBalance = EACConfig.BASEPATH + "datasets/balance/balance-scale.data"
    val filePathAdult = EACConfig.BASEPATH + "datasets/adult/adultCleaned.data"
    val filePathBC = EACConfig.BASEPATH + "datasets/breastcancer/bcCleaned.data"
    val filePathBankruptcy = EACConfig.BASEPATH + "datasets/bankruptcy/bankruptcy.data"
    val filePathCredit = EACConfig.BASEPATH + "datasets/credit/crxCleaned.data"
    val schemaStringAdult = "age workclass fnlwgt education education-num marital occupation relationship race sex capital-gain capital-loss hours-per-week country income"
    val schemaStringCar= "buying maint doors persons lug_boot safety acceptability"
    val schemaStringBalance = "class left-weight left-distance right-weight right-distance"
    val schemaStringBC = "clump_thickness u_cell_size u_cell_shape marginal_adhesion single_epithelial bare_nuclei bland_chromatin normal_nucleoli mitoses class"
    val schemaStringBankruptcy = "industrial_risk management_risk financial_flexibility credibility competitiveness operating_risk class"
    val schemaStringCredit = "a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16"
    //val readr= new carReader // new adultReader
    val readr = new CreditReader
    val indexed = readr.Indexed(filePathCredit /*filePathBalance*//*filePathCar*/ /*schemaStringBalance*/ /*schemaStringCar*/,sc)
    var transformed = readr.DFTransformed(indexed)
    val output = readr.Output(indexed)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val pw = new PrintWriter(new File("results_Balance.txt"))
    for (i <- 0 until 1) {
      val splits = transformed.randomSplit(Array(0.7, 0.3))
      val (trainingData, testData) = (splits(0), splits(1))
      val schema = StructType(readr.dataSchema.split(" ").zipWithIndex.map
        {case (fieldName, i) =>
      StructField(fieldName, DoubleType, true)})
      sqlContext.createDataFrame(trainingData.map(r => {
        val ro = r.features.toArray :+ r.label
        Row(ro.flatten)
      }), schema)

      //val numClasses = 4
      //val categoricalFeaturesInfo = Map[Int, Int]((0,4),(1,4),(2,4),(3,3),(4,3),(5,3))
      val numTrees = 100 // Use more in practice.
      val featureSubsetStrategy = "auto" // Let the algorithm choose.
      val impurity = "gini"
      val maxDepth = 5
      val maxBins = 32

      //trainingData.saveAsTextFile("train")
      //testData.saveAsTextFile("test")
      //val tmp: RDD[String] = sc.textFile(EACConfig.BASEPATH + "train.txt")
      //println(tmp.count().asInstanceOf[Int])
      //tmp.foreach(r => println(r.toString))
      //println("+++++++++++++++++++++++++++++++++++" + tmp.toString())

      val nfolds: Int = 20
      val knn = new EAC(7, 7, 7, trainingData, testData, readr.categoricalFeaturesInfo, readr.numericalFeaturesInfo)
      //val neighbors = testData.zipWithIndex().map{case (k, v) => (v, k)}
      //  .map(r => (r._1.asInstanceOf[Int], knn.getSortedNeighbors(r._2.features)))

      //neighbors.saveAsTextFile("neighbors")
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
        readr.numberOfClasses, readr.categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, 10)

      var boostingStrategy = BoostingStrategy.defaultParams("Classification")
      boostingStrategy.setNumIterations(3)
      boostingStrategy.treeStrategy.setNumClasses(2)
      boostingStrategy.treeStrategy.setMaxDepth(5)
      //boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

      //val gbModel = GradientBoostedTrees.train(trainingData, boostingStrategy)
      val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(readr.numberOfClasses).run(trainingData)
      val nbModel = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")
      val numIterations = 100
      //val svmModel = SVMWithSGD.train(trainingData, numIterations)
      //svmModel.clearThreshold()
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
      //val labelAndPreds = knn.getPredAndLabels()
      //println(testData.count().asInstanceOf[Int])
      val labeleAndPredsRF = testData.map {
        point => val prediction = model.predict(point.features)
          (point.label, prediction)
      }

      /*val labeleAndPredsGB = testData.map {
        point => val prediction = gbModel.predict(point.features)
          (point.label, prediction)
      }*/

      val labeleAndPredsLR = testData.map {
        point => val prediction = lrModel.predict(point.features)
          (point.label, prediction)
      }

      val labeleAndPredsNB = testData.map {
        point => val prediction = nbModel.predict(point.features)
          (point.label, prediction)
      }

      /*val labeleAndPredsSVM = testData.map{
      point => val prediction = svmModel.predict(point.features)
        (point.label, prediction)
    }*/

      val labelAndPreds = knn.getPredAndLabels()
      val labelAndPredKnn = knn.getPredAndLabelsKNN()
      //println(labelAndPreds)
      //println(labelAndPreds.filter(r => r._1 != r._2).count())
      val testErrKNN = labelAndPredKnn.filter(r => r._1 != r._2).length * 1.0 / testData.count()
      val testErr = labelAndPreds.filter(r => r._1 != r._2).length * 1.0 / testData.count()
      val testErrRF = labeleAndPredsRF.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0 / testData.count()
      //val testErrGB = labeleAndPredsGB.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0 / testData.count()
      val testErrLR = labeleAndPredsLR.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0 / testData.count()
      val testErrNB = labeleAndPredsNB.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0 / testData.count()
      //val testErrSVM = labeleAndPredsSVM.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0/testData.count()
      println("EAC Test Error = " + testErr + " RF test error = " + testErrRF + " KNN test error = " + testErrKNN +
        "  Logistic Regression test error " + testErrLR
        + " Naive Bayes test error " + testErrNB /*+ " GB test error " + testErrGB + " SVM test error " + testErrSVM*/)

      pw.write(testErr + " " + testErrRF + " " + testErrKNN + " " + testErrLR + " " + testErrNB /*+ " " + testErrGB*/)
    }
    pw.close
    //val testErrRF = labeleAndPredsRF.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0/testData.count()
    //println(testErrRF)
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
