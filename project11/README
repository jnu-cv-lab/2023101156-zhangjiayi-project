# 实验 ：实现并比较 Sinusoidal Position Encoding 与 RoPE

## 实验目的
通过手动实现正弦位置编码（Sinusoidal Position Encoding）和旋转位置编码（Rotary Position Embedding, RoPE），理解两者在 Transformer 模型中的工作原理，并对比它们在位置信息注入方式上的差异，验证 RoPE 的相对位置性质，并解释其相较于简单加法编码的巧妙之处。

## 实验任务

### 1. 实现 sinusoidal position encoding
按照原始 Transformer 论文中的公式生成位置编码矩阵：
\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
\]
其中 \(pos\) 为位置索引，\(i\) 为维度索引，\(d\) 为总维度。

### 2. 实现二维向量旋转
编写函数实现二维平面内向量的旋转：给定向量 \((x, y)\) 和旋转角度 \(\theta\)，输出旋转后的向量 \((x\cos\theta - y\sin\theta, x\sin\theta + y\cos\theta)\)。

### 3. 实现高维 RoPE
将高维特征向量（维度必须为偶数）的每一对相邻维度视为一个二维向量，对每个二维向量施加旋转，旋转角度为 \(\theta = pos \cdot \theta_i\)，其中 \(\theta_i = 10000^{-2i/d}\)。最终输出旋转后的完整向量。

### 4. 对比 E+pos 和 RoPE 的输入方式
- **E+pos**：直接将位置编码加到词嵌入（Embedding）上：\(E_{pos} = X + PE\)。
- **RoPE**：不对词嵌入做加法，而是分别对 Query 和 Key 向量进行旋转变换：\(Q_{pos} = R_{pos} \cdot Q\)，\(K_{pos} = R_{pos} \cdot K\)，其中 \(R_{pos}\) 为旋转矩阵。

### 5. 用数值实验验证 RoPE 的相对位置性质
- 生成随机的内容向量（所有位置内容相同）或固定内容。
- 分别计算经过 RoPE 后不同位置对之间的点积。
- 观察点积是否仅依赖于两个位置的差值，而与绝对位置无关。

### 6. 说明为什么 RoPE 比简单的 E+pos 更巧妙
从以下角度分析：
- 绝对位置编码与加法混合导致内容与位置无法解耦。
- RoPE 通过旋转变换将位置信息融入内积，自然产生相对位置关系。
- RoPE 具备更好的外推能力。

