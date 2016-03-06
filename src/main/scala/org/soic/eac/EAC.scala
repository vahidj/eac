package org.soic.eac

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLWritable, Identifiable}
import org.apache.spark.sql.DataFrame

final class EAC(override val uid: String)
  extends ProbabilisticClassifier[Vector, EAC, EACModel]
  with EACParams with DefaultParamsWritable
{
  def this() = this(Identifiable.randomUID("eac"))
  override protected def train(dataset: DataFrame): EACModel = {

  }
  override def copy(extra: ParamMap): EAC = defaultCopy(extra)
}

object EAC {

}

final class EACModel private[ml] () extends ProbabilisticClassificationModel[Vector, EACModel]
with EACParams with MLWritable {}
