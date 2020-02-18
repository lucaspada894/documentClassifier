import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class NaiveBayes {
    //CSV FILES
    private static List<List<Integer>> train_data = new ArrayList<List<Integer>>();
    private static List<Integer> train_labels = new ArrayList<Integer>();
    private static List<List<Integer>> test_data = new ArrayList<List<Integer>>();
    private static List<Integer> test_labels = new ArrayList<Integer>();

    //mapping.txt
    private static List<List<String>> mapping = new ArrayList<List<String>>();
    //vocabulary.txt
    private static List<String> vocabulary = new ArrayList<String>();



    //
    private static int[] classGroupNums;
    private static int[] n;
    private static int[][] nk;
    private static double[] Pwj; 			//CLASS PRIOR
    private static double[][] PMLE; 		//MAX LIKELIHOOD ESTIMATOR
    private static double[][] PBE;			//BAYESIAN ESTIMATOR


    private static int debug = 0;
    private static int v;
    private static double overall_accuracy_mle;
    private static double overall_accuracy_pbe;
    private static int[] classifiedAsClassMle;
    private static int[] classifiedAsClassPbe;
    private static double[] groupAccuracyMle;
    private static double[] groupAccuracyPbe;
    private static int[][] confusionMatrixMle;
    private static int[][] confusionMatrixPbe;

    public static void main(String args[]) {

        String train_label_path = "20newsgroups/train_label.csv"; //args[0];
        String train_data_path = "20newsgroups/train_data.csv"; //args[1];
        String test_label_path = "20newsgroups/test_label.csv"; //args[2];
        String test_data_path = "20newsgroups/test_data.csv"; //args[3];
        String mapping_path = "20newsgroups/map.csv";
        String vocabulary_path = "20newsgroups/vocabulary.txt";

        //Training part. Includes reading the data, obtaining class priors, n_k, MLE and BE.
        dataReader(train_label_path, train_data_path, test_label_path, test_data_path,mapping_path,vocabulary_path);
        getPrior();
        getTotalWordsForAllClasses();
        getWordOccurrencePerClass();
        getPMLEandPBE();

        //Evaluation part on both training and teting sets.
        System.out.println("\nPERFORMANCE ON TRAINING DATA");
        classify(0);
        outPutStuff();
        System.out.println("\nPERFORMANCE ON TESTING DATA");
        classify(1);
        outPutStuff();

    }

    /**
     * Obtains class prior. Fills the array Pwj with the class priors.
     */
    private static void getPrior() {
        classGroupNums = new int[20];
        Pwj = new double[20];
        int groupNum = 0;
        //initializing to 0
        for (int i = 0; i < classGroupNums.length; i++) {
            classGroupNums[i] = 0;
        }
        for (int i = 0; i < train_labels.size(); i++) {
            groupNum = train_labels.get(i);
            classGroupNums[groupNum - 1] += 1;
        }
        for(int i = 0; i < Pwj.length; i++)
            Pwj[i] = (float)classGroupNums[i]/(float)train_labels.size();
    }


    /**
     * Finds total amount of words for the 20 classes. Fills the array n.
     */
    private static void getTotalWordsForAllClasses(){
        n = new int[20];
        for(int i = 0; i < train_data.size(); i++){
            int count = train_data.get(i).get(2);
            int docId = train_data.get(i).get(0) - 1;
            int labelId = train_labels.get(docId) - 1;
            n[labelId] += count;
        }
    }

    /**
     * Fills array nk: occurence of each word in all documents belonging to class wj.
     */
    private static void getWordOccurrencePerClass(){
        nk = new int[vocabulary.size()][20];
        for(int i = 0; i < train_data.size(); i++){
            int count = train_data.get(i).get(2);
            int docId = train_data.get(i).get(0) - 1;
            int labelId = train_labels.get(docId) - 1;
            int wordId = train_data.get(i).get(1) - 1;
            nk[wordId][labelId] += count;
        }
    }

    /**
     * Gets the MLE and BE estimators. Fills the arrays PMLE and PBE
     */
    private static void getPMLEandPBE(){
        PMLE = new double[vocabulary.size()][20];
        PBE = new double[vocabulary.size()][20];
        for(int i = 0; i < vocabulary.size(); i++){
            for(int j = 0; j < 20; j++){
                PMLE[i][j] = (double)nk[i][j]/(double)n[j];
                PBE[i][j] = (nk[i][j]+1.0)/(double)(n[j]+vocabulary.size());
            }
        }
    }

    /**
     * Simple output of requested things. Outputs the overall accuracy for each estimator, as well as
     * class-individual accuracy.
     */
    private static void outPutStuff(){
        System.out.println("Class Priors:");
        for(int i = 0; i < Pwj.length; i++)
            System.out.printf("%.4f, ",Pwj[i]);
        System.out.println("\n");

        System.out.println("Overall Accuracy MLE: " + overall_accuracy_mle);
        System.out.println("Overall Accuracy PBE: " + overall_accuracy_pbe);
        System.out.println("\n");

        for(int i = 0; i < 20; i++){

            System.out.println("MLE Group " + (i+1) + ": "+ groupAccuracyMle[i]);
            System.out.println("PBE Group " + (i+1) + ": "+ groupAccuracyPbe[i]);
            System.out.println("\n");
        }
        System.out.println("\nConfusion Matrix MLE:");
        for(int i = 0; i < 20; i++){
            System.out.println();
            for(int j = 0; j < 20; j++){
                int k = confusionMatrixMle[i][j];
                System.out.print(k);
                if(String.valueOf(k).length() == 3)
                    System.out.print(" ");
                else if(String.valueOf(k).length() == 2)
                    System.out.print("  ");
                else
                    System.out.print("   ");

            }
        }

        System.out.println("\n\nConfusion Matrix PBE:");

        for(int i = 0; i < 20; i++){
            System.out.println();
            for(int j = 0; j < 20; j++){
                int k = confusionMatrixPbe[i][j];
                System.out.print(k);
                if(String.valueOf(k).length() == 3)
                    System.out.print(" ");
                else if(String.valueOf(k).length() == 2)
                    System.out.print("  ");
                else
                    System.out.print("   ");
            }
        }
        System.out.println();
    }




    /**
     * Classified the documents. If setType == 0, it classifies the documents belonging to the train samples. Otherwise,
     * it classssifies classifies the documents belonging to the test samples. Calls classifyHelper
     * @param setType
     */
    private static void classify(int setType){
        List<List<Integer>> dataSet;
        List<Integer> labels;
        int totalDocumentNumber;

        if(setType == 0){
            labels = train_labels;
            dataSet = train_data;
            totalDocumentNumber = train_labels.size();
        }
        else {
            labels = test_labels;
            dataSet = test_data;
            totalDocumentNumber = test_labels.size();
        }
        classifyHelper(labels,dataSet,totalDocumentNumber);
    }

    /**
     * Does the heavy classification work with the given labels, dataSet, and number of documents in total.
     * Classifies the document as belonging to class Wj if argmax is highest for class Wj.
     * Calls getAccuracy and getConfusionMatrix.
     * @param labels
     * @param dataSet
     * @param totalDocumentNumber
     */
    private static void classifyHelper(List<Integer> labels, List<List<Integer>> dataSet, int totalDocumentNumber){
        double[][] mleSums = new double[totalDocumentNumber][20];
        double pbeSums[][] = new double[totalDocumentNumber][20];
        double[][] Mle = new double[totalDocumentNumber][20];
        double[][] Pbe = new double[totalDocumentNumber][20];
        double[][] argmax_in = new double[totalDocumentNumber][20];
        int[] classifiedDocumentsMle = new int[totalDocumentNumber];
        int[] classifiedDocumentsPbe = new int[totalDocumentNumber];

        for(int i = 0; i < 20; i++){
            ArrayList<Double> sumMle = new ArrayList<Double>();
            //iterate train_data
            for(int j = 0; j < dataSet.size(); j++){
                int docId = dataSet.get(j).get(0) - 1;
                //int label = train_labels.get(docId) - 1;
                int wordId = dataSet.get(j).get(1) - 1;
                double mle = PMLE[wordId][i];
                double pbe = PBE[wordId][i];
                double logMle;
                logMle = Math.log(mle);
                double logPbe = Math.log(pbe);
                double Nk = dataSet.get(j).get(2);
                double rightSideMle = Nk*logMle;
                double rightSidePbe = Nk*logPbe;
                mleSums[docId][i] += rightSideMle;
                pbeSums[docId][i] += rightSidePbe;
            }

            for(int j = 0; j < totalDocumentNumber; j++){
                Mle[j][i] = Math.log(Pwj[i]) + mleSums[j][i];
                Pbe[j][i] = Math.log(Pwj[i]) + pbeSums[j][i];
            }
        }

        //Get Argmax
        int classNum = 0;
        int docNum = 0;
        double maxNumMle = Double.MAX_VALUE*-1.0;
        double maxNumPbe = Double.MAX_VALUE*-1.0;
        for(int i = 0; i < totalDocumentNumber; i++){
            maxNumMle = Double.MAX_VALUE*-1.0;
            maxNumPbe = Double.MAX_VALUE*-1.0;
            for(int j = 0; j < 20; j++){
                if(Mle[i][j] > maxNumMle) {
                    maxNumMle = Mle[i][j];
                    classifiedDocumentsMle[i] = j;
                }
                if(Pbe[i][j] > maxNumPbe){
                    maxNumPbe = Pbe[i][j];
                    classifiedDocumentsPbe[i] = j;
                }
            }

        }
        testAccuracy(labels,classifiedDocumentsMle,classifiedDocumentsPbe);
        getConfusionMatrix(labels,classifiedDocumentsMle,classifiedDocumentsPbe);
    }

    /**
     * Tests the overall accuracy and per class accuracy for each of the MLE and BE.
     * @param labels
     * @param classifiedDocumentsMle
     * @param classifiedDocumentsPbe
     */
    private static void testAccuracy(List<Integer> labels, int[] classifiedDocumentsMle,int[] classifiedDocumentsPbe){
        overall_accuracy_mle = 0;
        overall_accuracy_pbe = 0;
        groupAccuracyMle = new double[20];
        groupAccuracyPbe = new double[20];
        classifiedAsClassMle = new int[20];
        classifiedAsClassPbe = new int[20];
        for(int i = 0; i < labels.size(); i++){
            if((labels.get(i)-1) == classifiedDocumentsMle[i])
                overall_accuracy_mle += 1;
            if((labels.get(i)-1) == classifiedDocumentsPbe[i])
                overall_accuracy_pbe += 1;
        }
        overall_accuracy_mle = overall_accuracy_mle/labels.size();
        overall_accuracy_pbe = overall_accuracy_pbe/labels.size();


        for(int j = 0; j < classifiedDocumentsMle.length; j++) {
            if (classifiedDocumentsMle[j] == labels.get(j)-1)
                classifiedAsClassMle[labels.get(j)-1] += 1;
            if (classifiedDocumentsPbe[j] == labels.get(j)-1)
                classifiedAsClassPbe[labels.get(j)-1] += 1;
        }


        for(int i = 0; i < 20; i++){
            groupAccuracyMle[i] = (double)classifiedAsClassMle[i]/(double)classGroupNums[i];
            groupAccuracyPbe[i] = (double)classifiedAsClassPbe[i]/(double)classGroupNums[i];
        }
    }

    /**
     * calculates and prints the confusion matrix. Each cell (i,j) of the matrix represents the number of
     * documents in group i that were predicted to be in group j.
     * @param labels
     * @param classifiedDocumentsMle
     * @param classifiedDocumentsPbe
     */
    private static void getConfusionMatrix(List<Integer> labels, int[] classifiedDocumentsMle, int[] classifiedDocumentsPbe) {
        confusionMatrixMle = new int[20][20];
        confusionMatrixPbe = new int[20][20];
        for (int i = 0; i < labels.size(); i++) {
            confusionMatrixMle[labels.get(i) - 1][classifiedDocumentsMle[i]] += 1;
            confusionMatrixPbe[labels.get(i) - 1][classifiedDocumentsPbe[i]] += 1;
        }
    }

    /**
     * Scans and prints data.
     * @param trainLabelPath
     * @param trainDataPath
     * @param testLabelPath
     * @param testDataPath
     * @param mappingPath
     * @param vocabularyPath
     */
    private static void dataReader(String trainLabelPath, String trainDataPath, String testLabelPath, String testDataPath, String mappingPath, String vocabularyPath) {

        // read train labels
        try(BufferedReader reader = new BufferedReader(new FileReader(trainLabelPath))){
            String temp;
            temp = reader.readLine();
            while(temp != null) {
                train_labels.add(Integer.parseInt(temp));
                temp = reader.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        catch (IOException e){
            System.out.println(e);
        }

        //read train data

        try(BufferedReader reader = new BufferedReader(new FileReader(trainDataPath));){
            String temp = reader.readLine();
            while(temp != null) {
                String[] vals = temp.split(",");
                train_data.add(Arrays.asList(Integer.parseInt(vals[0]),Integer.parseInt(vals[1]),Integer.parseInt(vals[2])));
                temp = reader.readLine();
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        catch (IOException e){
            System.out.println(e);
        }


        // read test labels
        try(BufferedReader reader = new BufferedReader(new FileReader(testLabelPath));){
            String temp;
            temp = reader.readLine();
            while(temp != null) {
                test_labels.add(Integer.parseInt(temp));
                temp = reader.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        catch (IOException e){
            System.out.println(e);
        }
        try(BufferedReader reader = new BufferedReader(new FileReader(testDataPath));){
            String temp = reader.readLine();
            while(temp != null) {
                String[] vals = temp.split(",");
                test_data.add(Arrays.asList(Integer.parseInt(vals[0]),Integer.parseInt(vals[1]),Integer.parseInt(vals[2])));
                temp = reader.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        catch (IOException e){
            System.out.println(e);
        }

        File vocabularyFile = new File(vocabularyPath);
        String word = "";
        try {
            Scanner sc = new Scanner(vocabularyFile);
            while(sc.hasNextLine()) {
                word = sc.nextLine();
                vocabulary.add(word);
            }
            sc.close();
        }catch(FileNotFoundException e) {
            e.printStackTrace();
        }

        try(BufferedReader br = new BufferedReader(new FileReader(mappingPath))){
            String temp;
            while((temp=br.readLine())!=null) {
                String[] vals = temp.split(",");
                mapping.add(Arrays.asList(vals));
            }
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

}


