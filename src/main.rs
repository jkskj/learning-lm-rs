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

fn main() {
    // story();
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut cache = llama.new_cache();
    let mut user;
    // What are some potential applications for quantum computing?
    // What is one plus one?
    // What is my question?
    // let input ="<|im_start|>system\nAssistant is a large language model trained by OpenAI.\n<|im_end|>\n<|im_start|>user\nWho were the founders of Microsoft?\n<|im_end|>\n<|im_start|>assistant\n";
    let mut input: String;
    let mut output_ids;
    loop {
        user = String::new();
        std::io::stdin().read_line(&mut user).expect("WTF!");
        println!("{}", user.trim());
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
