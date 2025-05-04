use ndarray::{Array2, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::seq::SliceRandom;
use std::error::Error;
mod data_loader;
use data_loader::load_data;
mod nn_tester;


//Represents elements of neural network and is used in initialization function
struct NeuralNetwork {
   w_input_hidden1: Array2<f32>,
   w_hidden1_hidden2: Array2<f32>,
   w_hidden2_output: Array2<f32>,
   learning_rate: f32,
}


impl NeuralNetwork {
   fn new(input_size: usize, hidden1_size: usize, hidden2_size: usize, output_size: usize, learning_rate: f32) -> Self {
       let w_input_hidden1 = Array2::random((input_size, hidden1_size), Uniform::new(-0.5, 0.5));
       let w_hidden1_hidden2 = Array2::random((hidden1_size, hidden2_size), Uniform::new(-0.5, 0.5));
       let w_hidden2_output = Array2::random((hidden2_size, output_size), Uniform::new(-0.5, 0.5));
//Creates new neural network with random weights bween layers and given learning rate, take the inputs of the Neural Network elements from above struct
       Self {
           w_input_hidden1,
           w_hidden1_hidden2,
           w_hidden2_output,
           learning_rate,
       }
   }
// Applies the sigmoid activation function element-wise to a 2D array
   fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
       x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
   }
// Computes the derivative of the sigmoid function element-wise for a 2D array
   fn sigmoid_derivative(x: &Array2<f32>) -> Array2<f32> {
       x * &(1.0 - x)
   }
// Applies the softmax function row-wise to a 2D array, normalizing each row to a probability distribution
   fn softmax(x: &Array2<f32>) -> Array2<f32> {
       let mut result = Array2::zeros(x.raw_dim());// Initialize an array to hold the softmax result
       for (mut row_out, row_in) in result.outer_iter_mut().zip(x.outer_iter()) {
           let max = row_in.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
           let exps: Vec<f32> = row_in.iter().map(|v| (*v - max).exp()).collect();
           let sum: f32 = exps.iter().sum();
           for (i, val) in row_out.iter_mut().enumerate() {
               *val = exps[i] / sum;// Normalize each value in the row
           }
       }
       result
   }
// Performs a forward pass through the neural networ and returns activations for each layer
   fn forward(&self, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
       let z1 = x.dot(&self.w_input_hidden1);// Compute the weighted sum for the first hidden layer
       let a1 = Self::sigmoid(&z1);// Apply the sigmoid activation function to the first hidden layer


       let z2 = a1.dot(&self.w_hidden1_hidden2);//Repeat above steps for second hidden layer
       let a2 = Self::sigmoid(&z2);


       let z3 = a2.dot(&self.w_hidden2_output);
       let a3 = Self::softmax(&z3);//Repeat above steps for second hidden layer


       (a1, a2, a3)// Return activations for each layer: hidden1, hidden2, and output
   }
// Performs the backward pass and updates the weights using backpropagation and the computed gradients
   fn backward(&mut self, x: &Array2<f32>, a1: &Array2<f32>, a2: &Array2<f32>, a3: Array2<f32>, target: &[usize]) {
       let mut y_true = Array2::<f32>::zeros(a3.raw_dim());// Initialize the one-hot encoded true labels array
       for (i, &label) in target.iter().enumerate() {
           y_true[[i, label]] = 1.0;
       }
    

       let delta_output = a3 - &y_true;// Compute the error (difference) between predicted and true output
       let delta_hidden2 = delta_output.dot(&self.w_hidden2_output.t()) * Self::sigmoid_derivative(a2);// Compute the error for the second hidden layer
       let delta_hidden1 = delta_hidden2.dot(&self.w_hidden1_hidden2.t()) * Self::sigmoid_derivative(a1);//Repeat above for first layer
//Update weights
       self.w_hidden2_output = &self.w_hidden2_output - &(a2.t().dot(&delta_output).mapv(|v| v * self.learning_rate));
       self.w_hidden1_hidden2 = &self.w_hidden1_hidden2 - &(a1.t().dot(&delta_hidden2).mapv(|v| v * self.learning_rate));
       self.w_input_hidden1 = &self.w_input_hidden1 - &(x.t().dot(&delta_hidden1).mapv(|v| v * self.learning_rate));
   }
// Makes predictions by performing a forward pass and returning the index of the maximum output value for each input row
   fn predict(&self, input: &Array2<f32>) -> Vec<usize> {
       let (_, _, output) = self.forward(input);// Perform forward pass to get the output layer activations
       output.outer_iter()// Iterate over each row of the output
           .map(|row| row // For each row, find the index of the maximum value
               .iter()
               .enumerate()
               .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
               .unwrap().0)
           .collect()// Collect the indices (predicted classes) into a vector
   }
// Calculates the accuracy of the model by comparing predictions to the true labels
   fn accuracy(&self, input: &Array2<f32>, target: &[usize]) -> f32 {
       let preds = self.predict(input);// Get predictions for the input data
       preds.iter().zip(target.iter())
           .filter(|(a, b)| a == b)
           .count() as f32 / target.len() as f32// Count the number of correct predictions
   }
}
// Computes and prints the accuracy for each class separately, showing the number of correct predictions and the total for each class
fn per_class_accuracy(preds: &[usize], targets: &[usize], class_names: &[&str]) {
   let num_classes = class_names.len();
   let mut correct = vec![0usize; num_classes];// Initialize a vector to store the number of correct predictions for each class
   let mut total = vec![0usize; num_classes];// Initialize a vector to store the total number of samples for each class


   for (&pred, &actual) in preds.iter().zip(targets.iter()) {// Iterate over each prediction and corresponding target
       if pred == actual {
           correct[actual] += 1;
       }
       total[actual] += 1;
   }


   println!("Class Accuracy:");// Print the header for class accuracy output
   for i in 0..num_classes {
       let acc = if total[i] > 0 {
           correct[i] as f32 / total[i] as f32
       } else {
           0.0
       };
       // Print the class name, number of correct predictions, total samples, and accuracy percentage
       println!("{:<20} → {:>4} / {:<4} = {:.2}%", class_names[i], correct[i], total[i], acc * 100.0);
   }
}
// Main function to train and evaluate a neural network model on the dataset loaded from CSV files
fn main() -> Result<(), Box<dyn Error>> {
   let (train_inputs, train_labels) = load_data("/Users/jacobfreedman/DS210/ds210_project/src/train.csv")?;
   let (test_inputs, test_labels) = load_data("/Users/jacobfreedman/DS210/ds210_project/src/test.csv")?;
// Initialize the neural network with specified input size, hidden layers, output size, and learning rate
   let mut nn = NeuralNetwork::new(562, 128, 64, 6, 0.05);
   let epochs = 3;
// Training loop over specified epochs
   for epoch in 0..epochs {
       println!("Epoch {}/{}", epoch + 1, epochs);
       let mut indices: Vec<usize> = (0..train_inputs.nrows()).collect();
       indices.shuffle(&mut rand::rng());
// Training loop over each training sample
       for &i in &indices {
           let input = train_inputs.slice(s![i..=i, ..]).to_owned();
           let label = vec![train_labels[i]];
           let (a1, a2, a3) = nn.forward(&input);
           nn.backward(&input, &a1, &a2, a3, &label);
       }
// Evaluate the model's accuracy on the test data
       let acc = nn.accuracy(&test_inputs, &test_labels);
       println!("Overall Accuracy: {:.2}%", acc * 100.0);


       let preds = nn.predict(&test_inputs);
       let class_names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"];
       per_class_accuracy(&preds, &test_labels, &class_names);
   }


   Ok(())// Return Ok if everything runs successfully
}