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

class EAC private(private var k: Int)
  extends Serializable with Logging {
  def setK(k: Int): EAC = {
    this.k = k
    this
  }

  def getK: Int = k
  private var data: RDD[LabeledPoint] = null
  private var dataWithIndex: RDD[(Long, LabeledPoint)] = null
  //each element in the list contains the distance between pairs of values of the corrsponding feature
  private var mizan = new util.ArrayList[util.HashMap[(Double, Double), Int]](data.first().features.size)
  //2D array, indices represent indices of elements in data, each element represents distances between the case represented by row and column
  private var distances = new util.HashMap[(Int, Int), Double]
  //2D array, indices represent indices of elements in data, each row represents cases in data sorted by their ascending distance to the corresponding case
  private var neighbors = Array.ofDim[Int](data.count().asInstanceOf[Int], data.count().asInstanceOf[Int] - 1)

  def msort(array: List[Int], baseIndex:Int): List[Int] = {
    val n = array.length/2
    if (n == 0) array
    else{
      def merge(array1: List[Int], array2: List[Int]): List[Int] = (array1, array2) match {
        case (Nil, array2) => array2
        case (array1, Nil) => array1
        case (x :: array11, y :: array21) =>
          if (distances.get(baseIndex, x) < distances.get(baseIndex,y)) x :: merge(array11, array2)
          else y :: merge(array1, array21)
      }
      val (left, right) = array splitAt(n)
      merge(msort(left,baseIndex), msort(right,baseIndex))
    }
  }
  def getNearestNeighbors(i:Int): Array[Int] ={
    var result = range(0,dataWithIndex.count().asInstanceOf[Int]).toList
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
    //the following while loop generates VDM between all possible pairs of values for all features in the domain
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

    //the following section generates distances between all pairs of cases in the underlying domain
    for (i <- 0 until dataWithIndex.count().asInstanceOf[Int]){
      for (j <- i + 1 until dataWithIndex.count().asInstanceOf[Int]){
        //I'll put the smaller element as the first element of the tuple.
        distances.put((i,j), getDistance(i,j))
      }
    }

    //the following section list nearest neighbors for all cases in the domain
    for (i <- 0 until dataWithIndex.count().asInstanceOf[Int]){
      for (j <- i + 1 until dataWithIndex.count().asInstanceOf[Int]){
        neighbors(i) = getNearestNeighbors(i)
      }
    }
    new EACModel(dataWithIndex, mizan, k)
  }
}

object EAC {
  def train(input: RDD[LabeledPoint]): EACModel = {
    new EAC(0).run(input)
  }
}

class EACModel private[spark] (trainingSet: RDD[(Long, LabeledPoint)], inputMizan: util.ArrayList[util.HashMap[(Double, Double), Int]], inputK: Int)
  extends ClassificationModel with Serializable with Saveable{
  private val dataWithIndex: RDD[(Long, LabeledPoint)] = trainingSet
  private val mizan = inputMizan
  private val k = inputK


  def getDistance(c1:Vector, c2:Vector): Double = {
    var distance: Double = 0
    var featureCounter = 0
    c1.toArray.foreach(f1 => {
      val f2 = c2.toArray(featureCounter)
      val smaller = Math.min(f1, f2)
      val greater = Math.max(f1,f2)
      if (mizan.get(featureCounter).containsKey(smaller, greater))
        distance = scala.math.pow(mizan.get(featureCounter).get((smaller, greater)), 2)
      featureCounter += 1
    })
    math.sqrt(distance)
  }

  def getTopNeighbors(t:Vector): List[Int] = {
    var result = List[(Int, Double)]()
    this.dataWithIndex.foreach(r => {
      val tempDist = getDistance(t, r._2.features)
      result = result ::: List((r._1.asInstanceOf[Int], tempDist))
    })
    result.sortBy(_._2).map(_._1).take(this.k)
  }

  override def predict(testData: RDD[Vector]): RDD[Double] = {
    testData.map(t => {
      getTopNeighbors(t).map(dataWithIndex.lookup(_)(0).label).groupBy(identity).maxBy(_._2.size)._1
    })
  }

  override def predict(testData: Vector): Double = {
    0
  }

  override protected def formatVersion: String = "1.0"

  override def save(sc: SparkContext, path: String): Unit = {
    println("test")
  }
}
