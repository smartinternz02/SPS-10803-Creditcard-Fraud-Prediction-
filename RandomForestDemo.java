package org.ml;

import tech.tablesaw.api.Table;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.io.IOException;
import weka.classifiers.trees.RandomForest;

public class RandomForestDemo
{
	
	public static Instances getDataSet(String fileName) throws IOException 
	{
		
		/**
		 * we can set the file i.e., loader.setFile("filename") to load the data
		 */	
		int classIdx = 1;
		/** the CSVLoader to load the CSV file */
		CSVLoader loader = new CSVLoader();
		/** load the training data */
		//loader.setSource(RandomForestDemo.class.getResourceAsStream("/" + fileName));
		/**
		 * we can also set the file like loader3.setFile(new
		 * File("test-confused.arff"));
		 */
		loader.setFile(new File(fileName));
		Instances dataSet = loader.getDataSet();
		/** set the index based on the data given in the CSV files */
		dataSet.setClassIndex(classIdx);
		return dataSet;
		
	}
	/**
	 * This method is used to process the input and return the statistics.
	 * 
	 * @throws Exception
	 */
	public static void main(String args[]) throws Exception
	{
		
		try 
		{
			Table fraud_data=Table.read().csv("C:\\Users\\A. SRINIDHI\\eclipse-workspace\\org.ml\\src\\main\\java\\org\\ml\\fraud_dataset.csv");
			System.out.println(fraud_data.shape());
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
		
		Instances trainingDataSet = getDataSet("C:\\Users\\A. SRINIDHI\\eclipse-workspace\\org.ml\\src\\main\\java\\org\\ml\\credit_fraud_traindataset.csv");
		Instances testingDataSet = getDataSet("C:\\Users\\A. SRINIDHI\\eclipse-workspace\\org.ml\\src\\main\\java\\org\\ml\\credit_fraud_testdataset.csv");
		
		RandomForest forest=new RandomForest();
		forest.setNumFeatures(10);
		forest.buildClassifier(trainingDataSet);
		/**
		 * train the algorithm with the training data and evaluate the
		 * algorithm with testing data
		*/
		Evaluation eval = new Evaluation(trainingDataSet);
		eval.evaluateModel(forest, testingDataSet);
		
		/** Print the algorithm summary */
		System.out.println("** Random Forest Evaluation with Datasets **");
		System.out.println(eval.toSummaryString());
		System.out.print("The expression for the input data as per alogorithm is : ");
		System.out.println(forest);
	}
}