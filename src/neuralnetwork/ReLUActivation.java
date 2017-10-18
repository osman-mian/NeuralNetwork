/*
 * Author: Osman Ali
 */
package neuralnetwork;

import org.apache.commons.math.linear.RealMatrix;


public class ReLUActivation implements ActivationFunction{

    @Override
    //Returns elementwise ReLU of each matrix element
    public RealMatrix getActivation(RealMatrix A) {
        
        int r = A.getRowDimension();
        int c = A.getColumnDimension();
        
        for(int i=0;i<r;i++)
        {
            for(int j=0;j<c;j++)
            {
                A.setEntry(i, j, ReLU(A.getEntry(i, j)) );
            }
        }
        
        
        return A;
    }

    @Override
    //Returns elementwise ReLU gradient of a matrix 
    //Gradient is simply given by 1 if A[i][j]>0 and 0 otherwise
    public RealMatrix getLocalGradient(RealMatrix A) {
        int r = A.getRowDimension();
        int c = A.getColumnDimension();
        
        RealMatrix gradientMat= A.copy();//create a deep copy
        
        for(int i=0;i<r;i++)
        {
            for(int j=0;j<c;j++)
            {
                gradientMat.setEntry(i, j, Gradient_ReLU(A.getEntry(i, j)) );
            }
        }
        
        return gradientMat;
    }
    
    
    private double ReLU(double v)
    {
        return v>0 ? v : 0;
    }
    
    private double Gradient_ReLU(double v)
    {
        return v>0 ? 1 : 0;
    }
    
    
}
