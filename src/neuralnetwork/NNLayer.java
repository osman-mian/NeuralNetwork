/*
   Author: Osman Ali
 */
package neuralnetwork;

import org.apache.commons.math.linear.RealMatrix;


public interface NNLayer {
    
    public RealMatrix Forward(RealMatrix X);
    public RealMatrix Backward(RealMatrix X,double learningRate);
    
    public int GetWidth();
    public void InitializeWeights(int r,int c);
    
}
