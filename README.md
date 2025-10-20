# ml_learning_note
random ML reading notes
1. Hidden Markov Model
    - youtube [playlist](https://www.youtube.com/playlist?list=PLix7MmR3doRo3NGNzrq48FItR3TDyuLCo)
2. deep RL(感觉有些吃力）
    - youtube [playlist](https://www.youtube.com/watch?v=2GwBez0D20A&list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0)
3. 自注意力机制
    - https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

自注意力机制的核心是让每个输入元素（如单词、token或像素）根据其与序列中其他元素的关系，计算一个加权表示。这种机制能够捕捉序列中远距离的依赖关系，而不像传统的循环神经网络（RNN）那样受限于顺序处理或固定上下文。

- **“自”注意力**：指的是每个输入元素会根据自身与其他所有元素的关联性，动态调整注意力权重。
- **动态关注**：模型可以根据任务需要，关注序列中任意位置的信息，而不是仅限于邻近的元素。

### **工作原理**
自注意力机制通过以下步骤计算：

#### （1）**输入表示**
假设输入是一个序列 $X = [x_1, x_2, ..., x_n]$，其中 $x_i$ 是每个元素的向量表示（例如，词嵌入）。每个 $x_i$ 是一个 $d$ 维向量。

#### （2）**计算查询、键和值（Query, Key, Value）**
自注意力机制引入了三个可学习的矩阵，用于将输入向量投影到不同的子空间：
- **查询向量（Query, Q）**：表示当前元素“询问”其他元素的信息。
- **键向量（Key, K）**：表示其他元素是否与当前元素相关。
- **值向量（Value, V）**：包含实际的信息内容。

这些向量通过线性变换得到：

$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$

其中，$W_Q$, $W_K$, $W_V$ 是可学习的权重矩阵，形状为$d \times d_k$（或 $d \times d_v$），$Q, K, V$ 的形状为 $n \times d_k$或$n \times d_v$。

#### （3）**计算注意力权重**
注意力权重是通过查询和键的点积计算的，衡量每个输入元素与其他元素的相关性：
$\text{Attention Scores} = \frac{QK^T}{\sqrt{d_k}}$
- $QK^T$  是形状为 $n \times n$ 的矩阵，表示每个元素对其他元素的“相似度”。
- 除以 $\sqrt{d_k}$ 是缩放点积注意力（Scaled Dot-Product Attention）的关键步骤，用于避免点积值过大导致梯度消失。

接着，通过 Softmax 函数将这些分数归一化为权重：
$\text{Attention Weights} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$

#### （4）**加权求和**
使用注意力权重对值向量 \( V \) 进行加权求和，得到每个输入元素的上下文表示：
$\text{Output} = \text{Attention Weights} \cdot V$
输出是一个新的序列 $Z = [z_1, z_2, ..., z_n]$，每个 $z_i$ 是输入序列中所有值的加权组合，反映了当前元素对其他元素的关注程度。

==> $\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$

![diag](./image/self-attention.png)

4. Transformer 资料
- https://courses.grainger.illinois.edu/ece537/fa2022/slides/lec23.pdf 课件
- https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
- https://jalammar.github.io/illustrated-transformer/
- https://zhuanlan.zhihu.com/p/338817680 （只有过程 没有说明为什么）
- https://hackmd.io/@abliu/BkXmzDBmr （台大的Li教授视频讲的有点散）
- http://nlp.seas.harvard.edu/annotated-transformer/