/*
   Author: Osman Ali
 */
package neuralnetwork;

import org.apache.commons.math.linear.RealMatrix;


public interface ActivationFunction {
    
    public RealMatrix getActivation(RealMatrix A);
    public RealMatrix getLocalGradient(RealMatrix A);
    
}
