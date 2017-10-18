/*
   Author: Osman Ali
 */
package neuralnetwork;

import org.apache.commons.math.linear.MatrixUtils;
import org.apache.commons.math.linear.RealMatrix;


public class Main {
    
    public static void main(String[] argv)
    {
        

       //<editor-fold defaultstate="collapsed" desc="Sample dataset initialized">
       double[][] matrixData = new double[2][3]; //2d data arranged as column vectors, 3 samples
       RealMatrix X;
       matrixData[0][0]=10;
       matrixData[1][0]=10;
       
       matrixData[0][1]=-10;
       matrixData[1][1]=-10;
       
       matrixData[0][2]=0;
       matrixData[1][2]=0;
       
       double[][] t = new double[3][3]; //3d ouput using 1 hot encoding arranged as column vectors, 3 samples
       RealMatrix y ;
       t[0][0]=1.0;
       t[1][0]=0.0;
       t[2][0]=0.0;
       
       t[0][1]=0.0;
       t[1][1]=0.0;
       t[2][1]=1.0;
       
       t[0][2]=0.0;
       t[1][2]=1.0;
       t[2][2]=0.0;
       //</editor-fold>
       X = MatrixUtils.createRealMatrix(matrixData);
       y = MatrixUtils.createRealMatrix(t);
       Dataset TrainData = new Dataset(X,y);
       
       //MetaData
       int numIterations =100000;
       int inputDimensions=2;
       double learningRate=0.1;
       NeuralNetworkMetaData metaData = new NeuralNetworkMetaData(numIterations,inputDimensions,learningRate);
       
       //Layers
       NNLayer l1 = new HiddenLayer(2, new SigmoidActivation());
       NNLayer l2 = new HiddenLayer(3, new SigmoidActivation());
       NNLayer l3 = new HiddenLayer(3, new SigmoidActivation());
       NNLayer output = new OutputLayer(3, new SigmoidActivation());
       
       //NN Object
       NeuralNetwork nn = new NeuralNetwork(TrainData,metaData);
       nn.AddLayer(l1);
       nn.AddLayer(l2);
       nn.AddLayer(l3);
       nn.AddLayer(output);                                                     //MUST ADD output layer in the END only!!!
        
       nn.Initialize();                                                         //initialize the weights in layers
       nn.Optimize();                                                           //run gradient descent
       
       print(nn.Predict(X));                                                    //predict on the same 3 examples
       
    }
    
    //DEBUGGING PURPOSE ONLY
    private static void print(RealMatrix x)
    {
       double[][] m = x.transpose().getData();
       
       for(double[] row : m)
       {
           for(double val: row)
               System.out.print(val+" ");
           System.out.println("");
       }
   }
    
}
