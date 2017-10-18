/*
 * Author: Osman Ali
 */
package neuralnetwork;

import org.apache.commons.math.linear.RealMatrix;


public class TanhActivation implements ActivationFunction{

    @Override
    //Returns elementwise Tanh of a matrix 
    public RealMatrix getActivation(RealMatrix A) {
        
        int r = A.getRowDimension();
        int c = A.getColumnDimension();
        
        for(int i=0;i<r;i++)
        {
            for(int j=0;j<c;j++)
            {
                double newEntry=Tanh(A.getEntry(i, j));
                A.setEntry(i, j, newEntry );
            }
        }
        
        
        return A;
    }

    @Override
    //Returns elementwise TanH gradient of a matrix 
    //Gradient is simply given by 1-tanh^2(A[i][j])
    public RealMatrix getLocalGradient(RealMatrix A) {
        int r = A.getRowDimension();
        int c = A.getColumnDimension();
        RealMatrix gradientMat= A.copy();//create a deep copy
        
        for(int i=0;i<r;i++)
        {
            for(int j=0;j<c;j++)
            {
                gradientMat.setEntry(i, j, Gradient_Tanh(A.getEntry(i, j)) );
            }
        }
        
        return gradientMat;
    }
    
    
    private double Tanh(double v)
    {
        return Math.tanh(v);
    }
    
    private double Gradient_Tanh(double v)
    {
        //return 1-tanh^2(v)
        return (1.0-Math.pow(v, 2));
    }
    
    
    
}