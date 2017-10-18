/*
 * Author: Osman Ali
 */
package neuralnetwork;

import org.apache.commons.math.linear.RealMatrix;


public class SigmoidActivation implements ActivationFunction{

    @Override
    //Returns elementwise sigmoid of a matrix 
    public RealMatrix getActivation(RealMatrix A) {
        
        int r = A.getRowDimension();
        int c = A.getColumnDimension();
        
        for(int i=0;i<r;i++)
        {
            for(int j=0;j<c;j++)
            {
                A.setEntry(i, j, Sigmoid(A.getEntry(i, j)) );
            }
        }
        
        return A;
    }

    @Override
    //Returns elementwise sigmoid gradient of a matrix 
    //Gradient is simply given by sigmoid(A[i][j]) * (1-sigmoid(A[i][j]))
    public RealMatrix getLocalGradient(RealMatrix A) 
    {
        int r = A.getRowDimension();
        int c = A.getColumnDimension();
        
        RealMatrix gradientMat= A.copy();//create a deep copy
        
        for(int i=0;i<r;i++)
        {
            for(int j=0;j<c;j++)
            {
                gradientMat.setEntry(i, j, Gradient_Sigmoid(A.getEntry(i, j)));
            }
        }
        
        return gradientMat;
    }
    
    
    private double Sigmoid(double v)
    {
        double value = 1.0 / (1 + Math.exp(-v));
        return value;
    }
    
    private double Gradient_Sigmoid(double v)
    {
        return (v * (1.0-v));
    }
    
    
    
}
