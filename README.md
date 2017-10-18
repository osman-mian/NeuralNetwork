# NeuralNetwork

A neural network implementation for java (Still In Progress)

This is a simple and intuitive implementation of Neural Networks in Java using the Apache Common Math library. 
I have tried to keep the code simple and intuitive for anyone who just wishes to use a neural network for their application 
without getting into finer level details. 


# Sample Usage

```java

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
       
```