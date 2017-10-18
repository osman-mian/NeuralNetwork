/*
   Author: Osman Ali
 */
package neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math.linear.RealMatrix;


public class NeuralNetwork {

   List<NNLayer> layers = new ArrayList<>();                                    //List of Layers
   Dataset data;                                                                //The Data to train on
   NeuralNetworkMetaData metaData;                                              //Metadeta and hyper parameters
   
   //<editor-fold defaultstate="collapsed" desc="Public Members">
   public NeuralNetwork(Dataset data,NeuralNetworkMetaData metaData)
   {
       this.data=data;
       this.metaData=metaData;
   }
   
   
   //Initialize random weights in all layers
   //Do some validation checks
   public void Initialize()
   {

       NNLayer output_layer= layers.get(layers.size()-1);//last year MUST always be output layer
       if(this.data.getY().getRowDimension()!=output_layer.GetWidth())
       {
           throw new IllegalArgumentException(
                                                "Dataset output does not have the same number of dimensions{"
                                                +this.data.getY().getRowDimension()
                                                +"} as the output layer{"
                                                +output_layer.GetWidth()
                                                +"}");
       }
       
       if(this.data.getX().getRowDimension()!=metaData.inputDimensions)
       {
           throw new IllegalArgumentException(
                                                "Dataset input does not have the same number of dimensions{"
                                                +this.data.getX().getRowDimension()
                                                +"} as the inputDim property of the Neural Network{"
                                                +metaData.inputDimensions
                                                +"}");
       }
       
       int oldDim=metaData.inputDimensions;

       for(NNLayer layer : layers)
       {
           layer.InitializeWeights(oldDim, layer.GetWidth());
           oldDim=layer.GetWidth();
       }
       
   }
   
   //Make a prediction using the current state of the layers
   public RealMatrix Predict(RealMatrix Example)
   {
       return ForwardPass(Example);
   }
   
   //Add a Layer to neural network
   public void AddLayer(NNLayer layer)
   {
       layers.add(layer);
   }
   
   //Helpful function if you intend to use crossvalidation or minibatch
   public void UpdateTrainingData(Dataset data)
   {
       this.data=data;
   }
   public void Optimize()
   {
       for(int i=0;i<metaData.numIterations;i++)
       {
           ForwardPass(data.X);
           BackwardPass();
       }
       System.out.println("Optimization Finished");
   }
    
   //</editor-fold>
   
   //<editor-fold defaultstate="collapsed" desc="Forward and Backprop">

   //Making a prediction using current state of the layers
   //for each layer carry out the sequence  Activation( W.T*X)
   //and pass the result to next layer
   private RealMatrix ForwardPass(RealMatrix toyExample)
   {
        RealMatrix tempResult=toyExample;
        
        for(NNLayer layer : layers)
        {
            tempResult=layer.Forward(tempResult);
        }
        
        return tempResult;
   }
   
   //Each layer simple calculates its backprop error 
   //and returns it for previous layer
   private void BackwardPass()
   {
       int s = layers.size()-1;

       RealMatrix tempResult=data.y;
       for(int i=s;i>=0;i--)
       {
           tempResult=layers.get(i).Backward(tempResult,metaData.learningRate);
       }
   }

   //</editor-fold>

   //<editor-fold defaultstate="collapsed" desc="Helper Functions, Debug only!">
   private void print(RealMatrix x)
   {
       double[][] m = x.getData();
       
       for(double[] row : m)
       {
           for(double val: row)
               System.out.print(val+" ");
           System.out.println("");
       }
   }
   //</editor-fold>
   
   
}

