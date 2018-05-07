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

object Project {

	def main(args: Array[String]): Unit = {  
			val conf = new SparkConf()
					.setAppName("Project")

			val sc = new SparkContext(conf)
			
			// Split the records
			val inputTrain = sc.textFile(args(0))            
			.map(line => line.split(","))
			
			
			val trainSplits = inputTrain.randomSplit(Array(0.33, 0.33, 0.33), seed = 11L)
			
			// Persist the data which is going to be used for splitting
			val trainData0 = sparseNoRotations(trainSplits(0)).persist()
			val trainData1 = sparseNoRotations(trainSplits(1)).persist()
			
      // Random forests
			val algorithm = Algo.Classification
			val maximumDepth = 30
			val treeCount = 100
			val impurity = Entropy
			val numClasses = 2
			val featureSubsetStrategy = "auto"
			val seed = 123917
			val maxBins = 128
			
			val modelrf1 = RandomForest.trainClassifier(trainData0,
			    new Strategy(algorithm, impurity, maximumDepth, numClasses, subsamplingRate = 1.0),
			    treeCount, featureSubsetStrategy, seed)
			
			trainData0.unpersist()
			
			// Boosted Trees
      val boostingStrategy = BoostingStrategy.defaultParams("Classification")
      boostingStrategy.numIterations = 100
      boostingStrategy.treeStrategy.numClasses = 2
      boostingStrategy.treeStrategy.maxDepth = 2
      boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

      val modelbt1 = GradientBoostedTrees.train(trainData1, boostingStrategy)
			
			trainData1.unpersist()
			
			// Processing data for training the third model
			val temp = trainSplits(2).map(_.map(_.toDouble))
			
			val zeroData = temp.filter(x => x.last == 0)
			val oneData = temp.filter(x => x.last == 1)
			
			
			val zeroDataSample = zeroData.randomSplit(Array(0.125))
			val merged = zeroDataSample(0).union(oneData)
			
			
			val preprocData = merged.map(
          x => {x.zipWithIndex.map(_.swap).toSeq}
          )
      
      val trainData2 = preprocData.map(x => LabeledPoint(x.last._2,
          Vectors.sparse(x.init.length, x.init.filter(v => v._2 > 20.0))))
          
          
      val boostingStrategy2 = BoostingStrategy.defaultParams("Classification")
      boostingStrategy2.numIterations = 50
      boostingStrategy2.treeStrategy.numClasses = 2
      boostingStrategy2.treeStrategy.maxDepth = 2
      boostingStrategy2.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

      val modelbt2 = GradientBoostedTrees.train(trainData2, boostingStrategy2)
						
			
			trainData2.unpersist()

			
		  // Load the test file and persist
			val inputTest = sc.textFile(args(2))
			.map(line => line.split(","))
			
			val testData = sparseNoRotations(inputTest).persist()
			
			// Make predictions for the test file for all 3 models
			val labeledPredictionsrf1 = testData.map {labeledPoint =>
			val predictions = modelrf1.predict(labeledPoint.features)
			(labeledPoint.label, predictions)}
			
			val labeledPredictionsbt1 = testData.map {labeledPoint =>
			val predictions = modelbt1.predict(labeledPoint.features)
			(labeledPoint.label, predictions)}
			
			val labeledPredictionsbt2 = testData.map {labeledPoint =>
			val predictions = modelbt2.predict(labeledPoint.features)
			(labeledPoint.label, predictions)}
			
			// Unpersist the data after making predictions
			testData.unpersist()
			
			// Find predictions based on majority vote
			val votedlabel = majorityvotefn(labeledPredictionsrf1,labeledPredictionsbt1,labeledPredictionsbt2,args(1))
					
			val metricrf1 = new MulticlassMetrics(labeledPredictionsrf1)
			val metricbt1 = new MulticlassMetrics(labeledPredictionsbt1)
			val metricbt2 = new MulticlassMetrics(labeledPredictionsbt2)
			val metricvt = new MulticlassMetrics(votedlabel)
			
			// Check accuracy of all models
			val accuracyrf1 = metricrf1.accuracy
			val accuracybt1 = metricbt1.accuracy
			val accuracybt2 = metricbt2.accuracy
			val accuracyvt = metricvt.accuracy

			// Build a confusion matrix for all models
			val cfmatrf1 =  metricrf1.confusionMatrix
			val cfmatbt1 =  metricbt1.confusionMatrix
			val cfmatbt2 =  metricbt2.confusionMatrix
			val cfmatvt = metricvt.confusionMatrix
      
			// Save all models
			modelrf1.save(sc, args(3)+"/RF1")
			modelbt1.save(sc, args(3)+"/BT1")
			modelbt2.save(sc, args(3)+"/BT2")
			
			// Save the predictions for each model and the final majority model
			labeledPredictionsrf1.saveAsTextFile(args(1)+"/RF1")
			labeledPredictionsbt1.saveAsTextFile(args(1)+"/BT1")
			labeledPredictionsbt2.saveAsTextFile(args(1)+"/BT2")
			votedlabel.saveAsTextFile(args(1)+"/VOTE")
			
			val acc = Array(accuracyrf1, accuracybt1, accuracybt2, accuracyvt)
			val accRDD = sc.parallelize(acc, 1)
			accRDD.saveAsTextFile(args(1)+"/Accuracy")
			val cfmat = List(cfmatrf1, cfmatbt1, cfmatbt2, cfmatvt)
			val cfmatRDD = sc.parallelize(cfmat, 1)
			cfmatRDD.saveAsTextFile(args(1)+"/CFMAT")
			
			sc.stop()
	}
	
	
	def preprocWithoutNormalization(inputData: RDD[Array[String]]): RDD[LabeledPoint] = {
	 
		  val data = inputData.map(_.map(_.toDouble))

			val preprocData = data.map(x => LabeledPoint(x.last,
			    Vectors.dense(x.init)))
			    
			return preprocData
	}
	
	def preprocWithNormalization(inputData: RDD[Array[String]]): RDD[LabeledPoint] = {
	 
		  val data = inputData.map(_.map(_.toDouble))

			val preprocData = data.map(x => LabeledPoint(x.last,
			    Vectors.dense(x.init.map(rec => rec/255))))
			    
			return preprocData
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
	
	def majorityvotefn(inputData1: RDD[(Double,Double)],inputData2: RDD[(Double,Double)], inputData3: RDD[(Double,Double)], pth: String): RDD[(Double,Double)] ={

			val featurelabels = inputData1.map(x => x._1)
					.zipWithUniqueId()
					.map(_.swap)

			val predictedlabel1 = inputData1.map(x => x._2)
			.zipWithUniqueId()
			.map(_.swap)

			val predictedlabel2 = inputData2.map(x => x._2)
			.zipWithUniqueId()
			.map(_.swap)

			val predictedlabel3 = inputData3.map(x => x._2)
			.zipWithUniqueId()
			.map(_.swap)

			val voterdd = predictedlabel1.join(predictedlabel2)


			val voterdd2 = voterdd.join(predictedlabel3)
			.mapValues(f => Array(f._1._1,f._1._2,f._2))  
			.mapValues(f => {
				if(f.sum == 3)                
					1.0
				else if (f.sum == 2)
					0.0
				else
					0.0                                             
			})

			val votedlabel = voterdd2.join(featurelabels)
			.map(f => f._2)

			voterdd2.saveAsTextFile(pth+"/MVL")

			return votedlabel 
	}
	
		def sparseRotations(inputData: RDD[Array[String]]): RDD[LabeledPoint] = {
	 
      val data = inputData.map(_.map(_.toDouble))
      
      val mergedData = allRotations(data)
      
      val preprocData = mergedData.map(
          x => {x.zipWithIndex.map(_.swap).toSeq}
          )
      
      val preprocData2 = preprocData.map(x => LabeledPoint(x.last._2,
          Vectors.sparse(x.init.length, x.init.filter(v => v._2 > 20.0))))
			    
			return preprocData2
	}
		
		def allRotations(inputData: RDD[Array[Double]]): RDD[Array[Double]] ={
		  
		  val data = inputData.filter(x => x.last == 1)
		  
		  val rot90 = data.map(x => (rotate90(x.init):+x.last).toArray)
      val rot180 = data.map(x => (rotate180(x.init):+x.last).toArray)
      val rot270 = data.map(x => (rotate270(x.init):+x.last).toArray)
      val rot90Mirr = data.map(x => (rotate90(mirror(x.init)):+x.last).toArray)
      val rot180Mirr = data.map(x => (rotate180(mirror(x.init)):+x.last).toArray)
      val rot270Mirr = data.map(x => (rotate270(mirror(x.init)):+x.last).toArray)
      val mirr = data.map(x => (mirror_buf(x.init):+x.last).toArray)
      
      val mergedData = inputData.union(rot90).union(rot180).union(mirr).union(rot90Mirr).union(rot180Mirr).union(rot270).union(rot270Mirr)

      return mergedData
		}
		
		def rotate90(inputData: Array[Double]): ArrayBuffer[Double] = {
		  var rotation = ArrayBuffer.fill[Double](inputData.length)(0)
		  var i = 0
		  for (i <- 0 to inputData.length-1){
		    val z = floor(i/441).toInt; val x = floor(i/21).toInt; val y = i%21.toInt
		    rotation(z*21*21 + y*21 + (20-x)) = inputData(i)
		  }
		  return rotation
		}
		
		def rotate180(inputData: Array[Double]): ArrayBuffer[Double] = {
		  var rotation = ArrayBuffer.fill[Double](inputData.length)(0)
		  var i = 0
		  for (i <- 0 to inputData.length-1){
		    val z = floor(i/441).toInt; val x = floor(i/21).toInt; val y = i%21.toInt
		    rotation(z*21*21 + (20-x)*21 + (20-y)) = inputData(i)
		  }
		  return rotation
		}
		
		def rotate270(inputData: Array[Double]): ArrayBuffer[Double] = {
		  var rotation = ArrayBuffer.fill[Double](inputData.length)(0)
		  var i = 0
		  for (i <- 0 to inputData.length-1){
		    val z = floor(i/441).toInt; val x = floor(i/21).toInt; val y = i%21.toInt
		    rotation(z*21*21 + (20-y)*21 + (20-x)) = inputData(i)
		  }
		  return rotation
		}
		
		def mirror(inputData: Array[Double]): Array[Double] = {
		  var rotation = Array.fill[Double](inputData.length)(0)
		  var i = 0
		  for (i <- 0 to inputData.length-1){
		    val z = floor(i/441).toInt; val x = floor(i/21).toInt; val y = i%21.toInt
		    rotation(z*21*21 + (20-x)*21 + y) = inputData(i)
		  }
		  return rotation
		}
		def mirror_buf(inputData: Array[Double]): ArrayBuffer[Double] = {
		  var rotation = ArrayBuffer.fill[Double](inputData.length)(0)
		  var i = 0
		  for (i <- 0 to inputData.length-1){
		    val z = floor(i/441).toInt; val x = floor(i/21).toInt; val y = i%21.toInt
		    rotation(z*21*21 + (20-x)*21 + y) = inputData(i)
		  }
		  return rotation
		}
}
