/*
 * Author: Osman Ali
 */
package neuralnetwork;

//Class to store metaData and hyper parameters at one place
public class NeuralNetworkMetaData {
    
    public final int numIterations;                                             //number of epochs to make during training
    public final int inputDimensions;                                           //the size of the input dimensions
    public final double learningRate;                                           // the "eagerness" to reach the local optimum
    
    
    public NeuralNetworkMetaData(int numIterations,int inputDimensions,double learningRate)
    {
        this.numIterations=numIterations;
        this.inputDimensions=inputDimensions;
        this.learningRate=learningRate;
    }
    
}
