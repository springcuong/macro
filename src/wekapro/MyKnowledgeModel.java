/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package wekapro;

import java.io.File;
import java.io.IOException;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.Resample;

/**
 *
 * @author Duc Anh
 */
public class MyKnowledgeModel {
    DataSource source;
    Instances dataset;
    String[] model_options;
    String[] data_options;
    Instances trainset;
    Instances testset;
    public MyKnowledgeModel() {
    }
    
    //Doc du lieu vao bo nho
    public MyKnowledgeModel(String filename,
                            String m_opts,
                            String d_opts) throws Exception {
        this.source = new DataSource(filename);
        this.dataset = source.getDataSet();
        if(m_opts != null){
            this.model_options = weka.core.Utils.splitOptions(m_opts);
        } 
        if(d_opts != null){
            this.data_options = weka.core.Utils.splitOptions(d_opts);
        }       
    } 
    
    public  Instances removeData (Instances orginalData) throws Exception{
        Remove remove = new Remove();
        remove.setOptions(data_options);
        remove.setInputFormat(orginalData);
        return Filter.useFilter(orginalData, remove);
                
    }
    
    public Instances convert2Binary(Instances originalData) throws Exception{
        NominalToBinary n2b = new NominalToBinary();
        n2b.setOptions(data_options);
        n2b.setBinaryAttributesNominal(true);
        n2b.setInputFormat(originalData);
        return Filter.useFilter(originalData, n2b);
    }
    
    //Chuyen doi Numeric2Nominal
    public Instances converData (Instances orginalData) throws Exception{
        NumericToNominal n2n = new NumericToNominal();
        n2n.setOptions(data_options);
        n2n.setInputFormat(orginalData);
        return Filter.useFilter(orginalData, n2n);
    }

    //Xuat du lieu ra file
    public  void saveData(String filename) throws IOException{
        ArffSaver outData = new ArffSaver();
        outData.setInstances(this.dataset);
        outData.setFile(new File(filename));
        outData.writeBatch();
        System.out.println("Finished");
    }
    
    //Chuyen doi dinh dang du lieu
    public  void saveData2CSV(String filename) throws IOException{
         CSVSaver outData = new  CSVSaver();
         outData.setInstances(this.dataset);
         outData.setFile(new File(filename));
         outData.writeBatch();
         System.out.println("Converted");
        
    }
    
    public Instances divideTrainTest(Instances originalSet,
            double percent, boolean isTest) throws Exception{
        RemovePercentage rp = new RemovePercentage();
        rp.setPercentage(percent);
        rp.setInvertSelection(isTest);
        rp.setInputFormat(originalSet);
        return Filter.useFilter(originalSet, rp);
    }
    
    public Instances diviTrainTestR(Instances originalSet,
            double percent, boolean isTest) throws Exception{
        Resample rs = new Resample();
        rs.setNoReplacement(true);
        rs.setSampleSizePercent(percent);
        rs.setInvertSelection(isTest);
        rs.setInputFormat(originalSet);
        return Filter.useFilter(originalSet, rs);
    }
    
    public void saveModel(String filenam, Object model) throws Exception{
        weka.core.SerializationHelper.write(filenam, model);
    }
    public Object loadModel(String filename) throws Exception{
        return weka.core.SerializationHelper.read(filename);
    }
    @Override
    public String toString() {
        return  dataset.toSummaryString();
    }
    
}
