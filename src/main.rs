use ndarray::{Array2, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::seq::SliceRandom;
use std::error::Error;
use csv::ReaderBuilder;

fn load_data(path: &str) -> Result<(Array2<f32>, Vec<usize>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    
    let mut inputs = Vec::new();
    let mut labels = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let feature_vals: Vec<f32> = record.iter()
            .take(record.len() - 1)
            .map(|v| v.parse::<f32>().unwrap())
            .collect();

        let label_str = &record[record.len() - 1];
        let label = match label_str {
            "WALKING" => 0,
            "WALKING_UPSTAIRS" => 1,
            "WALKING_DOWNSTAIRS" => 2,
            "SITTING" => 3,
            "STANDING" => 4,
            "LAYING" => 5,
            _ => panic!("Unknown label: '{}'", label_str),
        };

        inputs.extend_from_slice(&feature_vals);
        labels.push(label);
    }

    let num_features = inputs.len() / labels.len();
    let features = Array2::from_shape_vec((labels.len(), num_features), inputs)?;
    Ok((features, labels))
}



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

        Self {
            w_input_hidden1,
            w_hidden1_hidden2,
            w_hidden2_output,
            learning_rate,
        }
    }

    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn sigmoid_derivative(x: &Array2<f32>) -> Array2<f32> {
        x * &(1.0 - x)
    }

    fn softmax(x: &Array2<f32>) -> Array2<f32> {
        let mut result = Array2::zeros(x.raw_dim());
        for (mut row_out, row_in) in result.outer_iter_mut().zip(x.outer_iter()) {
            let max = row_in.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row_in.iter().map(|v| (*v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (i, val) in row_out.iter_mut().enumerate() {
                *val = exps[i] / sum;
            }
        }
        result
    }

    fn forward(&self, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let z1 = x.dot(&self.w_input_hidden1);
        let a1 = Self::sigmoid(&z1);

        let z2 = a1.dot(&self.w_hidden1_hidden2);
        let a2 = Self::sigmoid(&z2);

        let z3 = a2.dot(&self.w_hidden2_output);
        let a3 = Self::softmax(&z3);

        (a1, a2, a3)
    }

    fn backward(&mut self, x: &Array2<f32>, a1: &Array2<f32>, a2: &Array2<f32>, a3: Array2<f32>, target: &[usize]) {
        let mut y_true = Array2::<f32>::zeros(a3.raw_dim());
        for (i, &label) in target.iter().enumerate() {
            y_true[[i, label]] = 1.0;
        }

        let delta_output = a3 - &y_true;
        let delta_hidden2 = delta_output.dot(&self.w_hidden2_output.t()) * Self::sigmoid_derivative(a2);
        let delta_hidden1 = delta_hidden2.dot(&self.w_hidden1_hidden2.t()) * Self::sigmoid_derivative(a1);

        self.w_hidden2_output = &self.w_hidden2_output - &(a2.t().dot(&delta_output).mapv(|v| v * self.learning_rate));
        self.w_hidden1_hidden2 = &self.w_hidden1_hidden2 - &(a1.t().dot(&delta_hidden2).mapv(|v| v * self.learning_rate));
        self.w_input_hidden1 = &self.w_input_hidden1 - &(x.t().dot(&delta_hidden1).mapv(|v| v * self.learning_rate));
    }

    fn predict(&self, input: &Array2<f32>) -> Vec<usize> {
        let (_, _, output) = self.forward(input);
        output.outer_iter()
            .map(|row| row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0)
            .collect()
    }

    fn accuracy(&self, input: &Array2<f32>, target: &[usize]) -> f32 {
        let preds = self.predict(input);
        preds.iter().zip(target.iter())
            .filter(|(a, b)| a == b)
            .count() as f32 / target.len() as f32
    }
}

fn per_class_accuracy(preds: &[usize], targets: &[usize], class_names: &[&str]) {
    let num_classes = class_names.len();
    let mut correct = vec![0usize; num_classes];
    let mut total = vec![0usize; num_classes];

    for (&pred, &actual) in preds.iter().zip(targets.iter()) {
        if pred == actual {
            correct[actual] += 1;
        }
        total[actual] += 1;
    }

    println!("Class Accuracy:");
    for i in 0..num_classes {
        let acc = if total[i] > 0 {
            correct[i] as f32 / total[i] as f32
        } else {
            0.0
        };
        println!("{:<20} → {:>4} / {:<4} = {:.2}%", class_names[i], correct[i], total[i], acc * 100.0);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let (train_inputs, train_labels) = load_data("/Users/jacobfreedman/DS210/ds210_project/src/train.csv")?;
    let (test_inputs, test_labels) = load_data("/Users/jacobfreedman/DS210/ds210_project/src/test.csv")?;

    let mut nn = NeuralNetwork::new(562, 128, 64, 6, 0.05);
    let epochs = 3;

    for epoch in 0..epochs {
        println!("Epoch {}/{}", epoch + 1, epochs);
        let mut indices: Vec<usize> = (0..train_inputs.nrows()).collect();
        indices.shuffle(&mut rand::rng());

        for &i in &indices {
            let input = train_inputs.slice(s![i..=i, ..]).to_owned();
            let label = vec![train_labels[i]];
            let (a1, a2, a3) = nn.forward(&input);
            nn.backward(&input, &a1, &a2, a3, &label);
        }

        let acc = nn.accuracy(&test_inputs, &test_labels);
        println!("Overall Accuracy: {:.2}%", acc * 100.0);

        let preds = nn.predict(&test_inputs);
        let class_names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"];
        per_class_accuracy(&preds, &test_labels, &class_names);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
