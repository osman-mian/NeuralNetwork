/*
 * Author: Osman Ali
 */
package neuralnetwork;

import java.util.HashMap;
import java.util.Random;
import org.apache.commons.math.linear.MatrixUtils;
import org.apache.commons.math.linear.RealMatrix;


public class Dataset {
    
    
    final RealMatrix X;//the samples
    final RealMatrix y;//the labels
    final int RowCount;//number of Samples in this set
    
    
    //to create a dataset from raw data of doubles matrices X and y
    //EACH ROW is 1 record, columns are the dimensions
    public Dataset(double[][] X,double[][] y) throws Exception
    {
        
        int xSamples = X.length;
        int ySamples = y.length;
        
        if(xSamples!=ySamples)
            throw new IllegalArgumentException("Dataset does not have same number of samples{"+xSamples+"} and outputs{"+ySamples+"}");
        
        
        this.RowCount=xSamples;
        this.X = MatrixUtils.createRealMatrix(X).transpose();
        this.y = MatrixUtils.createRealMatrix(y).transpose();
    }
    

    //to create a dataset from precreated matrices X and y
    //EACH COLUMN is 1 record. Records arranged as column vectors
    public Dataset(RealMatrix X,RealMatrix y)
    {
        int xSamples = X.getColumnDimension();
        int ySamples = y.getColumnDimension();
        
        if(xSamples!=ySamples)
            throw new IllegalArgumentException("Dataset does not have same number of samples{"+xSamples+"} and outputs{"+ySamples+"}");
        
        this.RowCount=xSamples;
        this.X=X.copy();
        this.y=y.copy();
    }
    
    
    
    
    //Generate a sequence of unique random indices, lenght same as the length of dataset request
    //Pick the rows pointed to by the indices and create a newDataset from them
    //return the new Dataset
    public Dataset SampleRandomly(int numSamples) throws Exception
    {
        if(numSamples>=this.RowCount)//return the same dataset if the required size is same or bigger
        {
            return new Dataset(this.X,this.y);
        }
            
        double[][] shuffledX = new double[numSamples][this.X.getRowDimension()];
        double[][] shuffledy = new double[numSamples][this.y.getRowDimension()];
        
        
        //to store the random indice sequence
        //each key is 1 index, so uniquencess guaranteed
        //TODO: make this approach efficient when numSamples is almost same as the current data size
        HashMap<Integer,Integer> sequence=new HashMap <>();
        
        Random r = new Random(System.nanoTime());
        int nextInt;
        
        //Generate a unique random sequence
        while(sequence.size()!=numSamples)
        {
            nextInt= r.nextInt();
            nextInt= nextInt % this.RowCount;
            nextInt= nextInt > 0 ? nextInt: -nextInt;
            
            if(!sequence.containsKey(nextInt))
            {
                sequence.put(nextInt, 1);
            }
        }
        
        //pick the generated indices samples from current dataset
        int counter=0;
        for(Integer v : sequence.keySet())
        {
            shuffledX[counter]=this.X.getColumn(v);
            shuffledy[counter]=this.y.getColumn(v);
            counter++;
        }
        
        //Create new dataset and return
        return new Dataset(shuffledX, shuffledy);
    }
    
    public RealMatrix getX()
    {
        return this.X;
    }
    
    public RealMatrix getY()
    {
        return this.y;
    }
    
}
