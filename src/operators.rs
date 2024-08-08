use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    // 获取维度数
    let ndim = y.shape().len();
    // 确保至少有2个维度
    assert!(ndim >= 2);
    // 序列的数量
    let seq_len = y.shape()[ndim - 2];
    // 每个序列的长度
    let total_seq_len = y.shape()[ndim - 1];
    // 批次数量
    let batch = y.size() / (seq_len * total_seq_len);
    // 获取数据的可变引用
    let data = unsafe { y.data_mut() };
    // 遍历每个批次
    for b in 0..batch {
        // 当前批次的基索引
        let base = b * seq_len * total_seq_len;
        // 遍历批次中的每个序列
        for i in 0..seq_len {
            // 当前序列的偏移量
            let offset = base + i * total_seq_len;
            // 软最大化的边界
            let boundary = total_seq_len - seq_len + i + 1;

            // 找到当前序列中最大值（直到边界）
            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            // 计算指数和归一化所需的和
            let sum = (0..boundary)
                .map(|j| {
                    // 指数值减去最大值
                    let e = (data[offset + j] - max).exp();
                    // 将指数值存回张量
                    data[offset + j] = e;
                    // 返回指数值
                    e
                })
                // 指数值的总和
                .sum::<f32>();
            // 通过除以和来归一化指数值，得到软最大化的概率
            (0..boundary).for_each(|j| data[offset + j] /= sum);
            // 将序列的其余部分设置为0
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    assert!(y.size() == x.size());
    // 获取维度数
    let ndim = y.shape().len();
    // 确保至少有2个维度
    assert!(ndim >= 2);
    // 序列的数量
    let seq_len = y.shape()[ndim - 2];
    // 每个序列的长度
    let total_seq_len = y.shape()[ndim - 1];
    // 获取维度数
    let wdim = w.shape().len();
    // 确保只有1个维度
    assert!(wdim == 1);
    // 确保长度相同
    assert!(w.size() == total_seq_len);
    // 批次数量
    let batch = y.size() / (seq_len * total_seq_len);
    // 获取数据的引用
    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    let _w = w.data();
    // 遍历每个批次
    for b in 0..batch {
        // 当前批次的基索引
        let base = b * seq_len * total_seq_len;
        // 遍历批次中的每个序列
        for l in 0..seq_len {
            // 当前序列的偏移量
            let offset = base + l * total_seq_len;
            // 平方和
            let s: f32 = _x[offset..offset + total_seq_len]
                .iter()
                .map(|f| f * f)
                .sum();
            let sqrt = (s / total_seq_len as f32 + epsilon).sqrt();
            // 计算并储存结果
            for i in 0..total_seq_len {
                _y[offset + i] = _w[i] * _x[offset + i] / sqrt;
            }
        }
    }
}

pub fn sigmoid(x: f32) -> f32 {
    let e = std::f32::consts::E;
    1. / (1. + e.powf(-x))
}
// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    for i in 0..len {
        _y[i] = sigmoid(_x[i]) * _x[i] * _y[i];
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    assert!(a.shape().len() == b.shape().len());
    assert!(a.shape().len() == c.shape().len());

    let ndim = a.shape().len();
    assert!(ndim >= 2);
    let a_seq_len = a.shape()[ndim - 2];
    let a_total_seq_len = a.shape()[ndim - 1];

    let b_seq_len = b.shape()[ndim - 2];
    let b_total_seq_len = b.shape()[ndim - 1];

    let c_seq_len = c.shape()[ndim - 2];
    let c_total_seq_len = c.shape()[ndim - 1];

    let _c = unsafe { c.data_mut() };
    let _a = a.data();
    let _b = b.data();

    assert!(a_total_seq_len == b_total_seq_len);
    assert!(c_total_seq_len == b_seq_len);
    assert!(a_seq_len == c_seq_len);

    for l in 0..c_seq_len {
        for i in 0..c_total_seq_len {
            let sum = (0..a_total_seq_len)
                .map(|j| _a[l * a_total_seq_len + j] * _b[i * b_total_seq_len + j])
                // 指数值的总和
                .sum::<f32>();
            _c[l * c_total_seq_len + i] = beta * _c[l * c_total_seq_len + i] + alpha * sum;
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
