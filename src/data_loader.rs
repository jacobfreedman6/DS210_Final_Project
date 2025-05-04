use ndarray::Array2;
use csv::ReaderBuilder;
use std::error::Error;


//Function for loading the CSV data into main
pub fn load_data(path: &str) -> Result<(Array2<f32>, Vec<usize>), Box<dyn Error>> {
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


