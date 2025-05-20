use tch::{Tensor, nn, Device, Kind};
use rand::Rng;
use std::vec::Vec;
use tch::nn::OptimizerConfig;
use std::path::Path;
use std::collections::HashMap;


const LOTTO_MAX: f64 = 49.0;  // Max lotto number
const NUM_NUMBERS: i64 = 6;   // Numbers per draw

#[derive(Debug)]
struct LottoNN {
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
}

impl LottoNN {
    fn new(vs: &nn::Path) -> LottoNN {
        let fc1 = nn::linear(vs, NUM_NUMBERS, 128, Default::default());
        let fc2 = nn::linear(vs, 128, 128, Default::default());
        let fc3 = nn::linear(vs, 128, NUM_NUMBERS, Default::default());
        LottoNN { fc1, fc2, fc3 }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.view([-1, NUM_NUMBERS])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .relu()
            .apply(&self.fc3)
            .sigmoid() * LOTTO_MAX  // Output scaled to [0, LOTTO_MAX]
    }

    fn save(&self, vs: &nn::VarStore, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        vs.save(path)?;
        Ok(())
    }

    fn load(vs: &mut nn::VarStore, path: &str) -> Result<LottoNN, Box<dyn std::error::Error>> {
        vs.load(path)?;
        Ok(LottoNN::new(&vs.root()))
    }
}

fn mse_loss(output: &Tensor, target: &Tensor) -> Tensor {
    let diff = output - target;
    diff.pow(2).mean(Kind::Float)
}

fn generate_random_lotto_data() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..500 {
        let input: Vec<f32> = (0..NUM_NUMBERS)
            .map(|_| rng.gen_range(0.0..LOTTO_MAX as f32))
            .collect();
        let target: Vec<f32> = (0..NUM_NUMBERS)
            .map(|_| rng.gen_range(0.0..LOTTO_MAX as f32))
            .collect();
        inputs.push(input);
        targets.push(target);
    }

    (inputs, targets)
}

fn predict_lotto_numbers(model: &LottoNN, current_state: Vec<f32>) -> Vec<u32> {
    let state_tensor = Tensor::of_slice(&current_state).view([1, NUM_NUMBERS]).to(Device::Cpu);
    let prediction = model.forward(&state_tensor);
    let values: Vec<f32> = Vec::<f32>::from(prediction.view([-1]));

    let mut lotto_numbers: Vec<u32> = values.iter()
        .map(|&x| x.round().clamp(1.0, LOTTO_MAX as f32) as u32)
        .collect();

    lotto_numbers.sort_unstable();
    lotto_numbers.dedup();

    let mut rng = rand::thread_rng();
    while lotto_numbers.len() < NUM_NUMBERS as usize {
        let new_num = rng.gen_range(1..=LOTTO_MAX as u32);
        if !lotto_numbers.contains(&new_num) {
            lotto_numbers.push(new_num);
        }
    }

    lotto_numbers.sort_unstable();
    lotto_numbers
}

fn train_model(
    vs: &nn::VarStore,
    model: &LottoNN,
    inputs: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
    num_epochs: i64
) -> f64 {
    let mut optimizer = nn::Adam::default().build(vs, 1e-3).unwrap();
    let mut final_loss = 0.0;

    for epoch in 0..num_epochs {
        //let mut total_loss = 0.0;
        let mut f_loss= 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let input_tensor = Tensor::of_slice(input).view([1, NUM_NUMBERS]).to(Device::Cpu);
            let target_tensor = Tensor::of_slice(target).view([1, NUM_NUMBERS]).to(Device::Cpu);

            let output = model.forward(&input_tensor);
            let loss = mse_loss(&output, &target_tensor);
            f_loss = f64::from(loss.detach());
            //total_loss += f_loss;


            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        //let avg_loss = total_loss / inputs.len() as f64;
        final_loss = f_loss; //avg_loss;

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, final_loss);
        }

        if epoch % 1000 == 0 || final_loss <= 0.01 {
            if let Err(e) = model.save(vs, "lotto_model.pt") {
                eprintln!("Failed to save model at epoch {}: {}", epoch, e);
            } else {
                println!("Model saved at epoch {}", epoch);
            }
        }

        if final_loss <= 0.01 {
            break;
        }
        
    }

    final_loss
}

fn generate_lotto_ticket(model: &LottoNN) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    let current_state: Vec<f32> = (0..NUM_NUMBERS)
        .map(|_| rng.gen_range(0.0..LOTTO_MAX as f32))
        .collect();
    predict_lotto_numbers(model, current_state)
}

fn main() {
    let epochs = 1000000;
    let model_file = "lotto_model.pt";

    let mut stats: HashMap<u32, u32> = HashMap::new();

    let (inputs, targets) = generate_random_lotto_data();

    let mut vs = nn::VarStore::new(Device::Cpu);

    let mut model: LottoNN = LottoNN::new(&vs.root());



    if Path::new(model_file).exists() {
        match LottoNN::load(&mut vs, model_file) {
            Ok(loaded_model) => {
                println!("Loaded existing model from '{}'", model_file);
                model = loaded_model;
            }
            Err(e) => {
                println!("Failed to load model: {}. Proceeding with new model.", e);
                let final_loss = train_model(&vs, &model, inputs.clone(), targets.clone(), epochs);
                println!("\nTraining completed with final loss: {:.6}", final_loss);
            }
        }
    }

    for _epoch in 0..epochs{
        let ticket = generate_lotto_ticket(&model);
        for ticket_number in ticket{
            *stats.entry(ticket_number).or_insert(0) += 1;
        }
    }

    println!("Most probable WINNING numbers have higher percentage:");
    for (number, value) in stats{
        println!("{}={}%", number, value as i64/epochs*100)
    }
}

