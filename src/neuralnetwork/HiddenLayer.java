/*
   Author: Osman Ali
 */
package neuralnetwork;

import org.apache.commons.math.linear.MatrixUtils;
import org.apache.commons.math.linear.RealMatrix;

public class HiddenLayer implements NNLayer{
    
    int width;                          //number of nodes in this layer
    ActivationFunction activation;      //what activation function will this layer use, only sigmoid available for now
    RealMatrix W;                       //the weights matrix
    
    
    
    RealMatrix result;                  // activation function applied to ( W.transpose * X) 
    RealMatrix localGradient;           //local gradient of the activation result
    RealMatrix previousActivations;     //output from previous layer, used in backprop
    
    public HiddenLayer(int width,ActivationFunction activation)
    {
        this.width=width;
        this.activation=activation;
    }
    
    @Override
    public int GetWidth()
    {
        return this.width;
    }
    
    @Override
    public void InitializeWeights(int r,int c) //Random Initialization
    {
        double[][] matrixData = new double[r][c];
        
        for(int i=0;i<r;i++)
        {
            for(int j=0;j<c;j++)
            {
                matrixData[i][j]=Math.random();
            }
        }
        
        W=MatrixUtils.createRealMatrix(matrixData);

    }
    
    @Override
    public RealMatrix Forward(RealMatrix X)
    {
        //take input
        //input*weight (+ bias later)
        //sigmoid for now
        //additionally keep the sigmoid gradient for backprop later
        //return the result
        
        previousActivations =X;
        result = activation.getActivation(W.transpose().multiply(X));
        localGradient = activation.getLocalGradient(result);
        
        return result;
    }
    
    @Override
    public RealMatrix Backward(RealMatrix inflowingResult,double learningRate)
    {
        int r=this.localGradient.getRowDimension();
        int c=this.localGradient.getColumnDimension();
        RealMatrix delta =  MatrixUtils.createRealMatrix(r,c);
        
        double res;
        for(int i=0;i<r;i++)
        {
            for(int j=0;j<c;j++)
            {
                res = inflowingResult.getEntry(i, j) * this.localGradient.getEntry(i, j);
                delta.setEntry(i, j, res);
            }
        }
        
        
        
        //update step
        RealMatrix newW =  
            W.subtract
            ( 
                previousActivations.multiply(delta.transpose()).scalarMultiply(learningRate)
            );
        
        RealMatrix outflowingResult = W.multiply(delta);                        //chain gradient to be sent to the previous layer in this backprop
        this.W = newW.copy();                                                  //update current weight
        
        return outflowingResult;                                                //return the result for the previous layer

    }
    
    
}
