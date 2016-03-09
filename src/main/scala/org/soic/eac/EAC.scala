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
import Array.range
import scala.collection._

class EAC(private var k: Int, data: RDD[LabeledPoint], testData: RDD[LabeledPoint])
  extends Serializable with Logging {
  def setK(k: Int): EAC = {
    this.k = k
    this
  }

  def getK: Int = k
  //private var data: RDD[LabeledPoint] = null
  private var dataWithIndex: RDD[(Long, LabeledPoint)] = data.zipWithIndex().map{case (k, v) => (v, k)}
  //each element in the list contains the distance between pairs of values of the corrsponding feature
  private var mizan = List.fill(this.data.first().features.size)(scala.collection.mutable.Map[(Double, Double),  Double]())
  //private var mizan = List[scala.collection.mutable.Map[(Double, Double), Double]]()//List[util.HashMap[(Double, Double), Int]]()
  //2D array, indices represent indices of elements in data, each element represents distances between the case represented by row and column
  private var distances = scala.collection.mutable.Map[(Int, Int), Double]()
  //2D array, indices represent indices of elements in data, each row represents cases in data sorted by their ascending distance to the corresponding case
  private var neighbors = Array.ofDim[Int](data.count().toInt, data.count().toInt - 1)

  def msort(array: List[Int], baseIndex:Int): List[Int] = {
    val n = array.length/2
    if (n == 0) array
    else{
      def merge(array1: List[Int], array2: List[Int]): List[Int] = (array1, array2) match {
        case (Nil, array2) => array2
        case (array1, Nil) => array1
        case (x :: array11, y :: array21) =>
          if (distances(baseIndex, x) < distances(baseIndex,y)) x :: merge(array11, array2)
          else y :: merge(array1, array21)
      }
      val (left, right) = array splitAt(n)
      merge(msort(left,baseIndex), msort(right,baseIndex))
    }
  }
  def getNearestNeighbors(i:Int): Array[Int] ={
    var result = range(0,data.count().asInstanceOf[Int]).toList
    msort(result, i).toArray[Int]
  }

  def getDistance(i:Long, j:Long): Double = {
    var distance: Double = 0
    val c1: LabeledPoint = this.dataWithIndex.lookup(i)(0)
    val c2: LabeledPoint = this.dataWithIndex.lookup(j)(0)
    var featureCounter = 0
    c1.features.toArray.foreach(f1 => {
      val f2 = c2.features.toArray(featureCounter)
      val smaller = Math.min(f1, f2)
      val greater = Math.max(f1,f2)
      if (mizan(featureCounter).contains(smaller, greater))
        distance = scala.math.pow(mizan(featureCounter)((smaller, greater)), 2)
      featureCounter += 1
    })
    math.sqrt(distance)
  }

  def getDistance(c1:Vector, c2:Vector): Double = {
    var distance: Double = 0
    var featureCounter = 0
    c1.toArray.foreach(f1 => {
      val f2 = c2.toArray(featureCounter)
      val smaller = Math.min(f1, f2)
      val greater = Math.max(f1,f2)
      if (mizan(featureCounter).contains(smaller, greater))
        distance = scala.math.pow(mizan(featureCounter)((smaller, greater)), 2)
      featureCounter += 1
    })
    math.sqrt(distance)
  }

  def getPredAndLabels(): List[(Double,Double)] = {
    var result = List[(Double,Double)]()
    testData.collect().foreach(point => result = result ::: List((point.label, predict(point.features))))
    result
  }

  def getTopNeighbors(t:Vector): List[Int] = {
    this.dataWithIndex.map(r => {
      (r._1.asInstanceOf[Int], getDistance(t, r._2.features))
    }).sortBy(_._2).map(_._1).take(this.k).toList
    /*var result = List[(Int, Double)]()
    this.dataWithIndex.foreach(r => {
      val tempDist = getDistance(t, r._2.features)
      println(tempDist)
      result = result ::: List((r._1.asInstanceOf[Int], tempDist))
    })
    result.sortBy(_._2).map(_._1).take(this.k)*/
  }

  def predict(testData: Vector): Double = {
    getTopNeighbors(testData).map(dataWithIndex.lookup(_)(0).label).groupBy(identity).maxBy(_._2.size)._1
  }

  def train(): EACModel = {
    val classStat = data.map(x => x.label).countByValue()
    var featureStat = List[Map[Double, Long]]()
    var featureClassStat = List[Map[(Double, Double), Long]]()
    for (i <- 0 until data.first().features.size){
      val tmp = data.map(x => x.features(i)).countByValue()
      featureStat =  featureStat ::: List(tmp)
      val tmp2 = data.map(x => (x.features(i), x.label)).countByValue()
      featureClassStat = featureClassStat ::: List(tmp2)
    }
    //println(featureClassStat.toString())
    null
    /*this.dataWithIndex = data.zipWithIndex().map{case (k,v) => (v, k)}
    //key: class value, value: how many records have this class value
    var classStat = scala.collection.mutable.Map[Double, Int]()
    //each element in the list is a hashmap with key: feature value, value: how many record have that value for the corresponding feature
    var featureStat = List.fill(data.first().features.size)(scala.collection.mutable.Map[Double, Int]())
    //each element in the list is a hashmap with key: tuple of feature value and class value, value: how many records match the key
    var featureClassStat = List.fill(data.first().features.size)(scala.collection.mutable.Map[(Double, Double), Int]())
    var test = 0
    mizan = List.fill(data.first().features.size)(scala.collection.mutable.Map[(Double, Double), Double]())
    println("=================================STARTED building featureStat and classStat=======================")
    data.foreach(r => {
      //featureStat.add(new util.HashMap[Double, Int])
      //featureClassStat.add(new util.HashMap[(Double, Double), Int])
      println(classStat.toString())
      println(test)
      test += 1

      var vectorIndex = 0
      if (classStat.contains(r.label)) {
        classStat(r.label) = classStat(r.label) + 1
      }
      else {
        classStat(r.label) = 1
      }
      r.features.toArray.foreach(f => {
        if (featureStat(vectorIndex).contains(f))
          featureStat(vectorIndex)(f) = featureStat(vectorIndex)(f) + 1
        else
          featureStat(vectorIndex)(f) = 1
        if (featureClassStat(vectorIndex).contains((f, r.label)))
          featureClassStat(vectorIndex)((f, r.label)) =  featureClassStat(vectorIndex)((f, r.label)) + 1
        else
          featureClassStat(vectorIndex)((f, r.label)) = 1
        vectorIndex += 1
      })
    })*/
    println("=================================STARTED building mizan=======================")
    //println(classStat.toString())
    //featureClassStat(0).keys.foreach(println(_))
    //var mizan2 = List.fill(data.first().features.size)(scala.collection.mutable.Map[(Double, Double),  Double]())
    val featureIt = featureStat.iterator
    var featureCounter = 0
    //the following while loop generates VDM between all possible pairs of values for all features in the domain
    while(featureIt.hasNext){
      //println("feature iterator")
      val featureValues = featureIt.next.keySet.toArray
      for (i <- 0 until featureValues.length){
        for (j <- i+1 until featureValues.length){
          val v1 = featureValues(i).asInstanceOf[Double]
          val v2 = featureValues(j).asInstanceOf[Double]
          val v1cnt = featureStat(featureCounter)(v1).toInt.toDouble
          val v2cnt = featureStat(featureCounter)(v2).toInt.toDouble
          var vdm = 0.0
          val classValsIt = classStat.keySet.iterator
          while(classValsIt.hasNext){
            val classVal = classValsIt.next()
            val tmp1 = featureClassStat(featureCounter).getOrElse(((v1, classVal)), 0L).toInt.toDouble
            val tmp2 = featureClassStat(featureCounter).getOrElse(((v2, classVal)), 0L).toInt.toDouble
            vdm += Math.abs( tmp1 / v1cnt -  tmp2 / v2cnt)
            //println(tmp1 + " " + tmp2 + " " + " " + tmp1 + " " +tmp2 +" "+ vdm)
          }
          //I'll put the smaller element as the first element of the tuple.
          //this makes looking up a tuple in mizan easier in future (meaning that if I want to check the
          // distance between two values, I'll always put the smaller value as the first element in the look up as well)

          //println(featureClassStat.toString())
          if (v1 <= v2) {
            mizan(featureCounter)((v1, v2)) = vdm
            //println(vdm)
          }
          else {
            mizan(featureCounter)((v2, v1)) = vdm
            //println(vdm)
          }
        }
      }
      featureCounter += 1
    }

    //println(mizan.toString())

    /*println("=================================STARTED building distances=======================")
    //the following section generates distances between all pairs of cases in the underlying domain
    for (i <- 0 until dataWithIndex.count().asInstanceOf[Int]){
      for (j <- i + 1 until dataWithIndex.count().asInstanceOf[Int]){
        //I'll put the smaller element as the first element of the tuple.
        distances.put((i,j), getDistance(i,j))
      }
    }

    println("=================================STARTED building neighbors=======================")
    //the following section list nearest neighbors for all cases in the domain
    for (i <- 0 until dataWithIndex.count().asInstanceOf[Int]){
      for (j <- i + 1 until dataWithIndex.count().asInstanceOf[Int]){
        neighbors(i) = getNearestNeighbors(i)
      }
    }*/
    println("IS ABOUT TO BUILD THE MODEL")
    new EACModel(k)
  }
}

/*object EAC {
  def train(input: RDD[LabeledPoint]): EACModel = {
    new EAC(0, input).run()
  }
}*/

class EACModel (inputK: Int)
  extends ClassificationModel with Serializable with Saveable{
  private val k = inputK



  override def predict(testData: RDD[Vector]): RDD[Double] = {
    testData.map(t => {
      predict(t)
    })
  }

  override def predict(testData: Vector): Double = {
    0.0
  }

  override protected def formatVersion: String = "1.0"

  override def save(sc: SparkContext, path: String): Unit = {
    println("test")
  }
}
