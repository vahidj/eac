package org.soic.eac

import java.util

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.Saveable
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkContext, SparkConf}
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLWritable, Identifiable}
import org.apache.spark.sql.{SQLContext, DataFrame}

class EAC private(private var lambda: Double)
  extends Serializable with Logging {
  def setLambda(lambda: Double): EAC = {
    this.lambda = lambda
    this
  }

  def getLambda: Double = lambda
  private var data: RDD[LabeledPoint] = null
  private var dataWithIndex: RDD[(Long, LabeledPoint)] = null
  //each element in the list contains the distance between pairs of values of the corrsponding feature
  private var mizan = new util.ArrayList[util.HashMap[(Double, Double), Int]](data.first().features.size)
  //2D array, indices represent indices of elements in data, each row represents cases in data sorted by their ascending distance to the corresponding case
  private var neighbors = Array.ofDim[Int](data.count().asInstanceOf[Int], data.count().asInstanceOf[Int] - 1)

  def getDistance(i:Long, j:Long): Double = {
    var distance: Double = 0
    val c1: LabeledPoint = this.dataWithIndex.lookup(i)(0)
    val c2: LabeledPoint = this.dataWithIndex.lookup(j)(0)
    var featureCounter = 0
    c1.features.toArray.foreach(f1 => {
      val f2 = c2.features.toArray(featureCounter)
      val smaller = Math.min(f1, f2)
      val greater = Math.max(f1,f2)
      if (mizan.get(featureCounter).containsKey(smaller, greater))
        distance = scala.math.pow(mizan.get(featureCounter).get((smaller, greater)), 2)
      featureCounter += 1
    })
    math.sqrt(distance)
  }

  def run(data: RDD[LabeledPoint]): EACModel = {
    this.data = data
    this.dataWithIndex = data.zipWithIndex().map{case (k,v) => (v, k)}
    //key: class value, value: how many records have this class value
    var classStat = new util.HashMap[Double, Int]()
    //each element in the list is a hashmap with key: feature value, value: how many record have that value for the corresponding feature
    var featureStat = new util.ArrayList[util.HashMap[Double, Int]](data.first().features.size)
    //each element in the list is a hashmap with key: tuple of feature value and class value, value: how many records match the key
    var featureClassStat = new util.ArrayList[util.HashMap[(Double, Double), Int]](data.first().features.size)


    data.foreach(r => {
      var vectorIndex = 0
      if (classStat.containsKey(r.label))
        classStat.put(r.label, classStat.get(r.label) + 1 )
      else
        classStat.put(r.label, 1 )
      r.features.toArray.foreach(f => {
        if (featureStat.get(vectorIndex).containsKey(f))
          featureStat.get(vectorIndex).put(f, featureStat.get(vectorIndex).get(f) + 1)
        else
          featureStat.get(vectorIndex).put(f, 1)
        if (featureClassStat.get(vectorIndex).containsKey((f, r.label)))
          featureClassStat.get(vectorIndex).put((f, r.label), featureClassStat.get(vectorIndex).get((f, r.label)) + 1)
        else
          featureClassStat.get(vectorIndex).put((f, r.label), 1)
        vectorIndex += 1
      })
    })

    val featureIt = featureStat.iterator()
    var featureCounter = 0
    val classValsIt = classStat.keySet().iterator()
    while(featureIt.hasNext){
      val featureValues = featureIt.next().keySet().toArray()
      for (i <- 0 until featureValues.length){
        for (j <- i+1 until featureValues.length){
          val v1 = featureValues(i).asInstanceOf[Double]
          val v2 = featureValues(j).asInstanceOf[Double]
          val v1cnt = featureStat.get(featureCounter).get(v1)
          val v2cnt = featureStat.get(featureCounter).get(v2)
          var vdm = 0
          while(classValsIt.hasNext){
            val classVal = classValsIt.next()
            vdm += Math.abs((featureClassStat.get(featureCounter).get((v1, classVal))/v1cnt) -
              (featureClassStat.get(featureCounter).get((v2, classVal))/v2cnt))
          }
          //I'll put the smaller element as the first element of the tuple.
          //this makes looking up a tuple in mizan easier in future (meaning that if I want to check the
          // distance between two values, I'll always put the smaller value as the first element in the look up as well)
          if (v1 <= v2)
            mizan.get(featureCounter).put((v1 , v2), vdm)
          else
            mizan.get(featureCounter).put((v2 , v1), vdm)
        }
      }
    }

    new EACModel(data)
  }
}

object EAC {
  def train(input: RDD[LabeledPoint]): EACModel = {
    new EAC(0).run(input)
  }
}

class EACModel private[spark] (trainingSet: RDD[LabeledPoint]) extends ClassificationModel with Serializable with Saveable{
  override def predict(testData: RDD[Vector]): RDD[Double] = {

    null
  }

  override def predict(testData: Vector): Double = {
    0
  }

  override protected def formatVersion: String = "1.0"

  override def save(sc: SparkContext, path: String): Unit = {
    println("test")
  }
}
