use crate::NeuralNetwork;
use crate::data_loader::load_data;
use ndarray::array;
use std::fs::File;
use std::io::Write;
use approx::assert_abs_diff_eq;


#[test]
fn test_load_data() -> Result<(), Box<dyn std::error::Error>> {
   let tmp_path = "test_sample.csv";
   let mut file = File::create(tmp_path)?;
   writeln!(file, "f1,f2,f3,label")?;
   writeln!(file, "1.0,2.0,3.0,WALKING")?;
   writeln!(file, "4.0,5.0,6.0,SITTING")?;


   let (features, labels) = load_data(tmp_path)?;
   assert_eq!(features.shape(), &[2, 3]);
   assert_eq!(labels, vec![0, 3]);


   std::fs::remove_file(tmp_path)?;
   Ok(())
}


#[test]
fn test_sigmoid_and_derivative() {
   let x = array![[0.0, 1.0], [-1.0, 2.0]];
   let sig = NeuralNetwork::sigmoid(&x);
   let deriv = NeuralNetwork::sigmoid_derivative(&sig);


   assert_abs_diff_eq!(sig[[0, 0]], 0.5, epsilon = 1e-6);
   assert!(deriv.iter().all(|v| *v >= 0.0 && *v <= 0.25));
}


#[test]
fn test_forward_shape() {
   let nn = NeuralNetwork::new(4, 5, 3, 2, 0.1);
   let input = array![[0.1, 0.2, 0.3, 0.4]];
   let (a1, a2, a3) = nn.forward(&input);


   assert_eq!(a1.shape(), &[1, 5]);
   assert_eq!(a2.shape(), &[1, 3]);
   assert_eq!(a3.shape(), &[1, 2]);
}


