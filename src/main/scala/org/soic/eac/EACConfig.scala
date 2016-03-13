package org.soic.eac

import com.typesafe.config.ConfigFactory

/**
  * Created by vjalali on 3/13/16.
  */
object EACConfig {
  val config = ConfigFactory.load("application.conf")
  lazy val BASEPATH = config.getString("eac.base.path")
}
