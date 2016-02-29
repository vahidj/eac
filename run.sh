mvn clean package
bash $SPARK_HOME/bin/spark-submit --class org.soic.eac.RF --master local[2] ./target/eac-0.0.1-SNAPSHOT.jar ./
