import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.Saveable
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import java.io.PrintWriter
import scala.collection.mutable.ListBuffer
import scala.math.floor
import scala.collection.mutable.ArrayBuffer


object Predictions {
	def main(args: Array[String]): Unit = {  
			val conf = new SparkConf()
					.setAppName("Eval")

			val sc = new SparkContext(conf)
 
			// Load the trained models
			val modelrf = RandomForestModel.load(sc, args(0) + "/RF1")
			val modelbt1 = GradientBoostedTreesModel.load(sc, args(0) + "/BT1")
			val modelbt2 = GradientBoostedTreesModel.load(sc, args(0) + "/BT2")
			
			val inputTest = sc.textFile(args(2))
			.map(line => line.split(","))
			
			// Pre-process the testdata
			val testData = sparseNoRotations(inputTest).persist()
			
			// Make predictions for each model
			val labeledPredictionsrf = testData.map {labeledPoint =>
			val predictions = modelrf.predict(labeledPoint.features)
			(labeledPoint.label, predictions)}
			
			val labeledPredictionsbt1 = testData.map {labeledPoint =>
			val predictions = modelbt1.predict(labeledPoint.features)
			(labeledPoint.label, predictions)}
			
			val labeledPredictionsbt2 = testData.map {labeledPoint =>
			val predictions = modelbt2.predict(labeledPoint.features)
			(labeledPoint.label, predictions)}
			
			// Unpersist the test data
			testData.unpersist()
			
			// Find predictions based on majority vote
			val votedlabel = majorityvotefn(labeledPredictionsrf, labeledPredictionsbt1, labeledPredictionsbt2)
			
			val metricrf = new MulticlassMetrics(labeledPredictionsrf)
			val metricbt1 = new MulticlassMetrics(labeledPredictionsbt1)
			val metricbt2 = new MulticlassMetrics(labeledPredictionsbt2)
			val metricvt = new MulticlassMetrics(votedlabel)
			
			// Check accuracy of all models
			val accuracyrf = metricrf.accuracy
			val accuracybt1 = metricbt1.accuracy
			val accuracybt2 = metricbt2.accuracy
			val accuracyvt = metricvt.accuracy
			
			// Build a confusion matrix for all models
			val cfmatrf =  metricrf.confusionMatrix
			val cfmatbt1 =  metricbt1.confusionMatrix
			val cfmatbt2 =  metricbt2.confusionMatrix
			val cfmatvt = metricvt.confusionMatrix
			
			// Save the predictions for each model and the final majority model
			val votedlabelOutput = votedlabel.map(x => x._2.toInt)
			votedlabelOutput.coalesce(1, false).saveAsTextFile(args(1)+"/VOTE")
			
			
			val acc = Array(accuracyrf, accuracybt1, accuracybt2, accuracyvt) 
			val accRDD = sc.parallelize(acc, 1)
			accRDD.saveAsTextFile(args(1)+"/Accuracy")
			val cfmat = List(cfmatrf, cfmatbt1, cfmatbt2, cfmatvt) 
			val cfmatRDD = sc.parallelize(cfmat, 1)
			cfmatRDD.saveAsTextFile(args(1)+"/CFMAT")
			
			sc.stop()
}

	def sparseNoRotations(inputData: RDD[Array[String]]): RDD[LabeledPoint] = {
	 
      val data = inputData.map(_.map(_.toDouble))
      
      val preprocData = data.map(
          x => {x.zipWithIndex.map(_.swap).toSeq}
          )
      
      val preprocData2 = preprocData.map(x => LabeledPoint(x.last._2,
          Vectors.sparse(x.init.length, x.init.filter(v => v._2 > 20.0))))
			    
			return preprocData2
	}

	def majorityvotefn(inputData1: RDD[(Double,Double)], inputData2: RDD[(Double,Double)], inputData3: RDD[(Double,Double)]): RDD[(Double,Double)] ={

			val featurelabels = inputData1.map(x => x._1)

			val predictedlabel1 = inputData1.map(x => x._2)

			val predictedlabel2 = inputData2.map(x => x._2)

			val predictedlabel3 = inputData3.map(x => x._2)

			val voterdd = predictedlabel1.zip(predictedlabel2)

			val voterdd1 = voterdd.zip(predictedlabel3)
			.map(f => ArrayBuffer(f._1._1,f._1._2,f._2))
			.map(f => {
				if(f.sum == 3)
					1.0
				else
					0.0
			})
			
			val votedlabel = featurelabels.zip(voterdd1)
			
			return votedlabel
	}
}