import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


// Start a simple Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

// Prepare training and test data.
val data_train = spark.read.option("header","true").option("inferSchema","true").format("csv").load("train.csv")
val data_test = spark.read.option("header","true").option("inferSchema","true").format("csv").load("test.csv")
data_train.printSchema()

////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
//////////////////////////////////////////////////

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Rename label column
// Grab only numerical columns
val df_train = data_train.select(data_train("Class").as("label"),$"V1",$"V2",$"V3",$"V4",$"V5",$"V6",$"V7",$"V8",
	$"V9",$"V10",$"V11",$"V12",$"V13",$"V14",$"V15",$"V16",$"V17",$"V18",$"V19",$"V20",$"V21",$"V22",$"V23",$"V24",
	$"V25",$"V26",$"V27",$"V28",$"normAmount")
val df_test = data_test.select(data_test("Class").as("label"),$"V1",$"V2",$"V3",$"V4",$"V5",$"V6",$"V7",$"V8",
	$"V9",$"V10",$"V11",$"V12",$"V13",$"V14",$"V15",$"V16",$"V17",$"V18",$"V19",$"V20",$"V21",$"V22",$"V23",$"V24",
	$"V25",$"V26",$"V27",$"V28",$"normAmount")

// An assembler converts the input values to a vector
// A vector is what the ML algorithm reads to train a model

// Set the input columns from which we are supposed to read the values
// Set the name of the column where the vector will be stored
val assembler_train = new VectorAssembler().setInputCols(Array("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","normAmount")).setOutputCol("features")
val assembler_test = new VectorAssembler().setInputCols(Array("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","normAmount")).setOutputCol("features")

// Transform the DataFrame
val output_train = assembler_train.transform(df_train).select($"label",$"features")
val output_test = assembler_test.transform(df_test).select($"label",$"features")

output_train.head()
//////////////////////////////////////
//////// Random Forrest Classification //////////
////////////////////////////////////
val rf = new RandomForestClassifier()
		//	.setLabelCol("label")
		//	.setFeaturesCol("features")
		//	.setNumTrees(10)

//////////////////////////////////////
/// PARAMETER GRID BUILDER //////////
////////////////////////////////////
val paramGrid = new ParamGridBuilder().addGrid(rf.numTrees,Array(10, 30, 100)).build()

///////////////////////
// TRAIN TEST SPLIT //
/////////////////////

// In this case the estimator is simply the linear regression.
// A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
// 80% of the data will be used for training and the remaining 20% for validation.
val trainValidationSplit = new TrainValidationSplit().setEstimator(rf).setEvaluator(new BinaryClassificationEvaluator()).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)


// You can then treat this object as the new model and use fit on it.
// Run train validation split, and choose the best set of parameters.
val model = trainValidationSplit.fit(output_train)

//////////////////////////////////////
// EVALUATION USING THE TEST DATA ///
////////////////////////////////////

// Make predictions on test data. model is the model with combination of parameters
// that performed best.
val results = model.transform(output_test).select("features", "label", "prediction")

results.show()

////////////////////////////////////
//// MODEL EVALUATION /////////////
//////////////////////////////////

// For Metrics and Evaluation
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Need to convert to RDD to use this
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instantiate metrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)