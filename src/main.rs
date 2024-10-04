mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;
fn story() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(input_ids, 500, 0.9, 4, 1.);
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

fn chat() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut cache = llama.new_cache();
    let mut user;
    let mut input: String;
    let mut output_ids;
    loop {
        user = String::new();
        std::io::stdin().read_line(&mut user).expect("WTF!");

        if user.trim().eq("quit") {
            break;
        }
        input = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user
        );
        // print!("{}", input);
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        (output_ids, cache) = llama.chat(input_ids, 500, 0.9, 4, 1., cache);
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    }
}

fn main() {
    println!("Enter 1 to call the story model, and enter 2 to call the chat model:");

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).expect("WTF!");

    match input.trim() {
        "1" => story(),
        "2" => chat(),
        _ => println!("Invalid input!"),
    }
}
