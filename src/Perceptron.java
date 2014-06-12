import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * You should implement your Perceptron in this class. 
 * Any methods, variables, or secondary classes could be added, but will 
 * only interact with the methods or variables in this framework.
 * 
 * You must add code for at least the 3 methods specified below. Because we
 * don't provide the weights of the Perceptron, you should create your own 
 * data structure to store the weights.
 * 
 */
public class Perceptron {

	/**
	 * The initial value for ALL weights in the Perceptron.
	 * We fix it to 0, and you CANNOT change it.
	 */
	public final double INIT_WEIGHT = 0.0;

	/**
	 * Learning rate value. You should use it in your implementation.
	 * You can set the value via command line parameter.
	 */
	public final double ALPHA;

	/**
	 * Training iterations. You should use it in your implementation.
	 * You can set the value via command line parameter.
	 */
	public final int EPOCH;
	//create weights variables, input units, and output units.
	double[] bias;
	double[][] weight;
	List<Double> input;
	double[] output;
	int numLabel;
	int numFeature;


	/**
	 * Constructor. You should initialize the Perceptron weights in this
	 * method. Also, if necessary, you could do some operations on
	 * your own variables or objects.
	 * 
	 * @param alpha
	 * 		The value for initializing learning rate.
	 * 
	 * @param epoch
	 * 		The value for initializing training iterations.
	 * 
	 * @param featureNum
	 * 		This is the length of input feature vector. You might
	 * 		use this value to create the input units.
	 * 
	 * @param labelNum
	 * 		This is the size of label set. You might use this
	 * 		value to create the output units.
	 */
	public Perceptron(double alpha, int epoch, int featureNum, int labelNum) {
		this.ALPHA = alpha;
		this.EPOCH = epoch;
		this.numLabel = labelNum;
		this.numFeature = featureNum;
		this.input = new ArrayList<Double>();
		this.output = new double[labelNum];
		this.bias = new double[labelNum];
		this.weight = new double[labelNum][featureNum];
		Arrays.fill(bias, INIT_WEIGHT);
		for(int i=0; i<weight.length; i++){
			Arrays.fill(weight[i], INIT_WEIGHT);
		}
	}

	/**
	 * Train your Perceptron in this method.
	 * 
	 * @param trainingData
	 */
	public void train(Dataset trainingData) {
		for(int i=0; i<EPOCH; i++){
			for(Instance ins:trainingData.instanceList){
				teach(ins);
			}
		}
	}
	
	/**
	 * Take in a training instance, predict class, and update weights
	 * @param trainIns the training instance we wish to teach the perceptron
	 */
	private void teach(Instance trainIns){
		this.input = trainIns.features;
		int actLab = Integer.parseInt(trainIns.label);
		prediction();
		update(actLab);
	}
	
	/**
	 * Predict classification of current input instance
	 * Needs the input instance to be in current perceptron's input variable
	 * @return classification as integer
	 */
	private int prediction(){
		//get wx linear combination of weights and x values (wx)
		double wx = 0;
		for(int lab=0; lab<numLabel; lab++){
			//add bias weight using bias term = +1
			wx = bias[lab];
			for(int feat=0; feat<numFeature; feat++){
				wx+=weight[lab][feat]*input.get(feat);
			}
			output[lab] = 1/(1+Math.exp(-wx));
		}
		//find max value index - the predicted classification
		double max = Double.NEGATIVE_INFINITY;
		int maxIndex=0;
		for(int i=0; i<numLabel; i++){
			if(output[i]>max){
				max = output[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	/**
	 * Updates the perceptron's weights according to the perceptron training rule
	 * Wi = Wi + Delta
	 * where Delta = ALPHA * ( actual - output) * output * ( 1 - output ) * Xi
	 * @param actLab the actual classification label (integer)
	 */
	private void update(int actLab){
		int actual;
		double delta;
		for(int lab=0; lab<numLabel; lab++){
			actual = (actLab==lab)?1:0;
			//calculate delta
			delta = ALPHA*(actual-output[lab])*output[lab]*(1-output[lab]);
			//bias weight update using bias term = +1
			bias[lab] += delta;
			//update all feature weights for this class label
			for(int feat=0; feat<numFeature; feat++){
				weight[lab][feat] += delta*input.get(feat);
			}
		}
	}
	

	/**
	 * Test your Perceptron in this method. Refer to the homework documentation
	 * for implementation details and requirement of this method.
	 * 
	 * @param testData
	 */
	public void classify(Dataset testData) {
		int miss = 0;
		int hit = 0;
			for(Instance ins:testData.instanceList){
				this.input = ins.features;
				int actLab = Integer.parseInt(ins.label);
				int pred = prediction();
				System.out.println(pred);
				if(actLab!=pred){
					miss++;
				}
				else{hit++;}
			}
		double acc = ((double) hit)/(hit+miss);
		//System.out.println("Hit: "+hit+"\nMiss: "+miss);
		System.out.printf("%.4f%n", acc);
	}

}