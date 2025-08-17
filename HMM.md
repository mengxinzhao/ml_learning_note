## 隐藏马尔可夫模型 (HMM) 例子：不诚实的赌场

### Overview
#### 例子背景

赌场有两种骰子（隐藏状态）

公平骰子 (Fair Die, 简称 F)：每个面（1 到 6）的概率均为 $ \frac{1}{6} \approx 0.1667 $。
作弊骰子 (Loaded Die, 简称 L)：面 1 到 5 的概率均为 $ \frac{1}{10} = 0.1 $，面 6 的概率为 $ \frac{1}{2} = 0.5 $（偏向出现 6）。


赌场会在两种骰子间随机切换（隐藏过程），但你只能看到掷出的点数序列（观察序列）。
目标：基于观察序列，推断隐藏状态（哪个骰子被使用）、计算序列概率，并在参数未知时从数据中估计参数。
示例观察序列：$ O = [3, 5, 1, 6, 6, 6, 2] $，长度 $ T = 7 $。
为计算方便，将观察映射到index：$ 1 \to 0, 2 \to 1, 3 \to 2, 4 \to 3, 5 \to 4, 6 \to 5 $。因此序列索引为 $ [2, 4, 0, 5, 5, 5, 1] $。

#### HMM 三大件

- 隐藏状态集合 (States)：$ S = \{F, L\} $，数量 $ N = 2 $。用索引表示：$ 0 = F, 1 = L $。
- 观察符号集合 (Observations)：$ V = \{1, 2, 3, 4, 5, 6\} $，数量 $ M = 6 $。
- 初始状态概率 (Start Probability, $ \pi $)：$ \pi(i) = P(q_1 = i) $，表示序列开始时状态 $ i $ 的概率。真实值：
$$\pi = \begin{bmatrix} 0.5 & 0.5 \end{bmatrix}$$


#### HMM 矩阵
- 转移概率矩阵 (Transition Probability, $ A $)：$ a_{ij} = P(q_{t+1} = S_j | q_t = S_i) $，每行和为 1。例子中骰子是掺假的：
$$A = \begin{bmatrix}
    0.95 & 0.05 \\
    0.10 & 0.90
\end{bmatrix}$$
Note： 转移概率矩阵是对于hidden state而言。

- 发射概率矩阵 (Emission Probability, $ B $)：$ b_j(k) = P(o_t = v_k | q_t = S_j) $，每行和为 1。例子观测值：
$$B = \begin{bmatrix}
    \frac{1}{6} & \frac{1}{6} & \frac{1}{6} & \frac{1}{6} & \frac{1}{6} & \frac{1}{6} \\
    0.1 & 0.1 & 0.1 & 0.1 & 0.1 & 0.5
\end{bmatrix}$$
Note： 发射概率矩阵是对于观测者而言。

观察序列 (Observation Sequence, $ O $)：$ O = (o_1, o_2, \ldots, o_T) $。

- HMM 模型记号：$ \lambda = (\pi, A, B) $。

#### HMM 假设

马尔可夫属性：$ P(q_{t+1} | q_1, \ldots, q_t) = P(q_{t+1} | q_t) $。
观察独立性：$ P(o_t | q_1, \ldots, q_T, o_1, \ldots, o_{t-1}, o_{t+1}, \ldots, o_T) = P(o_t | q_t) $。

### 计算步骤
#### 评估 (Evaluation): 计算观察序列的概率
目标：给定模型参数 $ \lambda $ 和观察序列 $ O $，计算 $ P(O | \lambda) $，即序列的似然度，用于评估模型拟合度。
算法：前向算法 (Forward Algorithm)（动态规划，避免指数计算）。

定义：前向概率 $ \alpha_t(i) = P(o_1, o_2, \ldots, o_t, q_t = S_i | \lambda) $。
步骤：

初始化：
$$\alpha_1(i) = \pi(i) \cdot b_i(o_1), \quad i = 1, 2$$

递推：
$$\alpha_t(j) = \left[ \sum_{i=1}^N \alpha_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(o_t), \quad t = 2, \ldots, T, \quad j = 1, 2$$

终止：
$$P(O | \lambda) = \sum_{i=1}^N \alpha_T(i)$$




在例子中的手动推导（使用真实参数，观察序列 $ [3, 5, 1, 6, 6, 6, 2] $，索引 $ [2, 4, 0, 5, 5, 5, 1] $）：

t=1, $ o_1 = 3 $ (索引 2)：
$$\alpha_1(1) = 0.5 \cdot \frac{1}{6} \approx 0.083333 \quad (F)$$
$$\alpha_1(2) = 0.5 \cdot 0.1 = 0.05 \quad (L)$$

t=2, $ o_2 = 5 $ (索引 4)：
$$\alpha_2(1) = \left[ 0.083333 \cdot 0.95 + 0.05 \cdot 0.10 \right] \cdot \frac{1}{6} \approx [0.079167 + 0.005] \cdot 0.166667 \approx 0.014028$$
$$\alpha_2(2) = \left[ 0.083333 \cdot 0.05 + 0.05 \cdot 0.90 \right] \cdot 0.1 \approx [0.004167 + 0.045] \cdot 0.1 \approx 0.004917$$

继续计算到 $ t=7 $，最终：
$$P(O | \lambda) = \alpha_7(1) + \alpha_7(2) \approx 1.52 \times 10^{-6}$$
（概率小，因为序列包含连续的 6，表明可能使用了作弊骰子）。
后向算法 (Backward Algorithm)（为后续学习问题准备）：

定义：$ \beta_t(i) = P(o_{t+1}, \ldots, o_T | q_t = S_i, \lambda) $。
初始化：
$$\beta_T(i) = 1, \quad i = 1, 2$$

递推：
$$\beta_t(i) = \sum_{j=1}^N a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j), \quad t = T-1, \ldots, 1$$




复杂度：$ O(N^2 T) $，高效避免了直接计算所有可能状态序列的指数复杂度。
步骤 3: 问题 2 - 解码 (Decoding): 找出最可能的隐藏状态序列
目标：给定 $ \lambda $ 和 $ O $，找出最可能的隐藏状态序列 $ Q = (q_1, \ldots, q_T) $，即：
$$Q^* = \arg\max_Q P(Q | O, \lambda)$$
算法：Viterbi 算法（动态规划，使用对数避免概率下溢）。

定义：

$ V_t(j) = \max_{q_1, \ldots, q_{t-1}} P(q_1, \ldots, q_{t-1}, q_t = S_j, o_1, \ldots, o_t | \lambda) $
路径变量：$ \psi_t(j) = \arg\max_{i=1}^N \left[ V_{t-1}(i) + \ln(a_{ij}) \right] $


步骤：

初始化：
$$V_1(i) = \ln \left( \pi(i) \cdot b_i(o_1) \right), \quad \psi_1(i) = 0, \quad i = 1, 2$$

递推：
$$V_t(j) = \max_{i=1}^N \left[ V_{t-1}(i) + \ln(a_{ij}) \right] + \ln(b_j(o_t)), \quad t = 2, \ldots, T$$
$$\psi_t(j) = \arg\max_{i=1}^N \left[ V_{t-1}(i) + \ln(a_{ij}) \right]$$

终止：
$$P^* = \max_{j=1}^N V_T(j), \quad q_T^* = \arg\max_{j=1}^N V_T(j)$$

回溯：从 $ q_T^* $ 开始，使用 $ \psi_t(j) $ 反向构建 $ Q^* $。



在例子中的手动推导（使用真实参数）：

t=1, $ o_1 = 3 $ (索引 2)：
$$V_1(1) = \ln \left( 0.5 \cdot \frac{1}{6} \right) \approx \ln(0.083333) \approx -2.4849$$
$$V_1(2) = \ln \left( 0.5 \cdot 0.1 \right) \approx \ln(0.05) \approx -2.9957$$
$$\psi_1(1) = \psi_1(2) = 0$$

t=2, $ o_2 = 5 $ (索引 4)：
$$V_2(1) = \max \left[ -2.4849 + \ln(0.95), -2.9957 + \ln(0.10) \right] + \ln \left( \frac{1}{6} \right)$$
$$\approx \max \left[ -2.4849 - 0.0513, -2.9957 - 2.3026 \right] - 1.7918 \approx \max[-2.5362, -5.2983] - 1.7918 \approx -4.3280$$
$$\psi_2(1) = 1 \quad (F \to F)$$
$$V_2(2) = \max \left[ -2.4849 + \ln(0.05), -2.9957 + \ln(0.90) \right] + \ln(0.1)$$
$$\approx \max \left[ -2.4849 - 2.9957, -2.9957 - 0.1054 \right] - 2.3026 \approx \max[-5.4806, -3.1011] - 2.3026 \approx -5.4037$$
$$\psi_2(2) = 2 \quad (L \to L)$$

继续到 $ t=7 $，最终最佳路径约为 $ Q^* = [F, F, F, L, L, L, F] $，对数概率 $ \ln(P^*) \approx -13.7 $。（连续 6 导致切换到 L）。

模拟结果（见步骤 4）显示估计参数下的路径为 $ ['F', 'F', 'L', 'L', 'L', 'L', 'L'] $。
步骤 4: 问题 3 - 学习/估计 (Learning/Estimation): 从观察序列估计参数
目标：给定观察序列 $ O $，在参数未知时，最大化 $ P(O | \lambda) $ 以估计 $ \lambda = (\pi, A, B) $。
算法：Baum-Welch 算法（期望最大化 EM 算法的特化版本，迭代直到收敛）。

概述：E-step 计算隐藏变量的期望，M-step 更新参数。
定义：

状态后验：
$$\gamma_t(i) = P(q_t = S_i | O, \lambda) = \frac{\alpha_t(i) \cdot \beta_t(i)}{P(O | \lambda)}$$

转移后验：
$$\xi_t(i,j) = P(q_t = S_i, q_{t+1} = S_j | O, \lambda) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{P(O | \lambda)}$$

$ P(O | \lambda) = \sum_{i=1}^N \alpha_T(i) $。


完整步骤：

初始化：随机设置 $ \pi, A, B $，确保归一化。例子中初始值：
$$\pi^{(0)} = \begin{bmatrix} 0.6 & 0.4 \end{bmatrix}$$
$$A^{(0)} = \begin{bmatrix}
    0.7 & 0.3 \\
    0.4 & 0.6
\end{bmatrix}$$
$$B^{(0)} = \begin{bmatrix}
    0.1 & 0.2 & 0.1 & 0.2 & 0.2 & 0.2 \\
    0.1 & 0.1 & 0.1 & 0.1 & 0.1 & 0.5
\end{bmatrix}$$

E-step：计算 $ \gamma_t(i) $ 和 $ \xi_t(i,j) $。

使用前向算法计算 $ \alpha_t(i) $。
使用后向算法计算 $ \beta_t(i) $。
计算 $ \gamma_t(i) $ 和 $ \xi_t(i,j) $。


M-step：更新参数：
$$\pi(i) = \gamma_1(i)$$
$$a_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$
$$b_j(k) = \frac{\sum_{t=1}^T \gamma_t(j) \cdot \mathbb{I}(o_t = v_k)}{\sum_{t=1}^T \gamma_t(j)}, \quad \mathbb{I} \text{为指示函数}$$

归一化 $ A $ 和 $ B $ 的每行和为 1。


迭代：重复 E 和 M 步，直到参数变化小于阈值 $ \epsilon = 0.001 $，或固定迭代次数（这里为 5 次）。



在例子中的应用：

使用 Python 模拟运行，观察索引 $ [2, 4, 0, 5, 5, 5, 1] $，迭代 5 次。
迭代 1 的 E-step 示例（手动简化展示，完整计算在模拟）：

计算 $ \alpha_t(i) $ 和 $ \beta_t(i) $。
对于 $ t=1, o_1=3 $ (索引 2)：
$$\alpha_1(1) = 0.6 \cdot 0.1 = 0.06, \quad \alpha_1(2) = 0.4 \cdot 0.1 = 0.04$$
计算 $ \beta_1(i) $ 需要整个序列，假设 $ P(O | \lambda) \approx 1.2 \times 10^{-5} $（模拟值）。
$$\gamma_1(1) \approx \frac{\alpha_1(1) \cdot \beta_1(1)}{P(O | \lambda)} \approx 0.639$$

$ \xi_t(i,j) $ 类似计算，关注连续 6 处 $ \xi_t(2,2) $（L 到 L）较高。


M-step：基于 $ \gamma $ 和 $ \xi $ 更新参数。
模拟结果（每步输出，完整计算）：

迭代 1：
$$\pi^{(1)} = \begin{bmatrix} 0.63923293 & 0.36076707 \end{bmatrix}$$
$$A^{(1)} = \begin{bmatrix}
    0.7 & 0.3 \\
    0.4 & 0.6
\end{bmatrix}$$
$$B^{(1)} = \begin{bmatrix}
    0.1573639 & 0.19188911 & 0.18837355 & 0.0 & 0.21003456 & 0.25233888 \\
    0.12920768 & 0.09672279 & 0.10003059 & 0.0 & 0.07964967 & 0.59438927
\end{bmatrix}$$

迭代 2：
$$\pi^{(2)} = \begin{bmatrix} 0.81277768 & 0.18722232 \end{bmatrix}$$
$$B^{(2)} = \begin{bmatrix}
    0.15674799 & 0.16671557 & 0.24281482 & 0.0 & 0.22647814 & 0.20724349 \\
    0.13012759 & 0.12099332 & 0.05125608 & 0.0 & 0.06622699 & 0.63139601
\end{bmatrix}$$

迭代 3：
$$\pi^{(3)} = \begin{bmatrix} 0.96734931 & 0.03265069 \end{bmatrix}$$
$$B^{(3)} = \begin{bmatrix}
    0.1587419 & 0.12631237 & 0.31324898 & 0.0 & 0.26408937 & 0.13760737 \\
    0.13031741 & 0.1559179 & 0.00834654 & 0.0 & 0.04715408 & 0.65826407
\end{bmatrix}$$

迭代 4：
$$\pi^{(4)} = \begin{bmatrix} 0.99953525 & 0.00046475 \end{bmatrix}$$
$$B^{(4)} = \begin{bmatrix}
    0.16827825 & 0.06167288 & 0.37981871 & 0.0 & 0.33011823 & 0.06011192 \\
    0.12754292 & 0.19176429 & 0.00010639 & 0.0 & 0.03004703 & 0.65053937
\end{bmatrix}$$

迭代 5：
$$\pi^{(5)} = \begin{bmatrix} 0.99999996 & 0.00000004 \end{bmatrix}$$
$$B^{(5)} = \begin{bmatrix}
    0.17693899 & 0.00972415 & 0.41573305 & 0.0 & 0.38316686 & 0.01443694 \\
    0.12501447 & 0.21255548 & 0.00000001 & 0.0 & 0.01704919 & 0.64538085
\end{bmatrix}$$

转移矩阵 $ A $：在模拟中逐步接近真实值 $ [0.95, 0.05; 0.1, 0.9] $，但由于序列短且初始值随机，5 次迭代后可能仍接近初始值。更多迭代会进一步收敛。
发射矩阵 $ B $：接近真实值，尤其是 $ b_2(6) \approx 0.645 $，反映作弊骰子偏向 6。


使用估计参数的 Viterbi 路径：
$$Q^* = ['F', 'F', 'L', 'L', 'L', 'L', 'L']$$
（前两个为公平骰子，后续因连续 6 切换到作弊骰子）。

总结
这个例子完整展示了 HMM 的三个核心问题：

评估：前向算法计算 $ P(O | \lambda) \approx 1.52 \times 10^{-6} $，表明序列可能包含作弊。
解码：Viterbi 算法找出最佳状态序列，反映连续 6 时切换到作弊骰子。
学习：Baum-Welch 算法从随机初始参数迭代估计 $ \lambda $，发射概率逐渐接近真实值，路径推断合理。
