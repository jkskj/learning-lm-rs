use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| match safetensor.tensor(name) {
            Ok(t) => Tensor::<f32>::new(
                (0..(t.data().len() / 4))
                    .into_iter()
                    .map(|f| f32::from_le_bytes(t.data()[(f * 4)..(f * 4 + 4)].try_into().unwrap()))
                    .collect(),
                &Vec::from(t.shape()),
            ),
            Err(e) => {
                println!("{},{}", name, e);
                let shape: Vec<usize> = Vec::new();
                Tensor::<f32>::default(&shape)
            }
        };
        let embedding_table = if config.tie_word_embeddings {
            "lm_head.weight"
        } else {
            "model.embed_tokens.weight"
        };
        let get_tensors = |name: &str| {
            (0..config.num_hidden_layers)
                .into_iter()
                .map(|i| {
                    let mut n: String = String::from("model.layers.");
                    n.push_str(&i.to_string());
                    n.push('.');
                    n.push_str(name);
                    get_tensor(&n)
                })
                .collect()
        };
        LLamaParams {
            embedding_table: get_tensor(embedding_table),
            rms_att_w: get_tensors("input_layernorm.weight"),
            wk: get_tensors("self_attn.k_proj.weight"),
            wo: get_tensors("self_attn.o_proj.weight"),
            wq: get_tensors("self_attn.q_proj.weight"),
            wv: get_tensors("self_attn.v_proj.weight"),
            rms_ffn_w: get_tensors("post_attention_layernorm.weight"),
            w_up: get_tensors("mlp.up_proj.weight"),
            w_down: get_tensors("mlp.down_proj.weight"),
            w_gate: get_tensors("mlp.gate_proj.weight"),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
