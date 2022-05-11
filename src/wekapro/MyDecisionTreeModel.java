/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package wekapro;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Debug.Random;
import weka.core.Instances;

/**
 *
 * @author Spring Cuong
 */
public class MyDecisionTreeModel extends MyKnowledgeModel{
    J48 tree;

    public MyDecisionTreeModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
    }
    
    public void buildDecisionTree() throws Exception{
        this.trainset = diviTrainTestR(this.dataset, 80, false);
        this.testset = diviTrainTestR(this.dataset, 80, true);
        this.trainset.setClassIndex(this.trainset.numAttributes() -1);
        this.testset.setClassIndex(this.testset.numAttributes() -1);
        tree = new J48();
        tree.setOptions(this.model_options);
        tree.buildClassifier(this.trainset);
    }
    
    public void evaluateDecisionTree() throws Exception{
        Random rnd = new Debug.Random(1);
        int folds = 10;
        Evaluation eval = new Evaluation(this.trainset);
        eval.crossValidateModel(tree, this.testset, folds, rnd);
        System.out.println(eval.toSummaryString("\nKet qua danh gia mo hinh 10-fold cross-validation\n--------\n", false));
    }
    
    public void predictClassLabel(Instances input) throws Exception{
        for(int i = 0; i < input.numInstances(); i++){
            double predict = tree.classifyInstance(input.instance(i));
            double actual = input.instance(i).classValue();
            System.err.println("Instance " + i + ": predict = " + input.classAttribute().value((int)predict) + "; actual = " + 
                    input.classAttribute().value((int)actual));
//            input.instance(i).setClassValue(predict);
        }
    }
    @Override
    public String toString() {
        return tree.toSummaryString();
    }
    
    
}
