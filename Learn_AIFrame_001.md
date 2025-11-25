# 1       矩阵论

1、 怎么理解矩阵的秩？

·矩阵的秩是告诉你这个矩阵中有多少行列相互独立的，不能通过其他行/列通过线性组合的方式得到。

·高秩意味着这个矩阵信息量更大，而低秩矩阵更容易压缩。

2、给你一个矩阵A，求这个矩阵的秩是多少？提示：使用“高斯消元法”

将矩阵A化简一个上三角或者下三角矩阵，然后看这个矩阵中非零行数量即为秩的结果。

3、如何计算exp(A) ，其中A为一个矩阵。

4、是否了解最小二乘法？能否简单讲述下最小二乘法的步骤。

5、讲解下矩阵(M*M)特征值分解的步骤

6、讲解下SVD奇异值分解。

 

# 2   概率论
1、 仅使用一个random(3)生成器，完成一个random(7)生成器(多种方式)。

2、 是否了解贝叶斯公式？

3、 讲一下最大似然估计，它的应用场景有哪些？

4、 讲一下KL散度和JS散度，它们的数学原理是什么？分别可以用到哪里？

 
# 3       离散数学/数论
1、 余弦相似度的公式。

2、 是否了解牛顿迭代法？能否把公式写下来？牛顿迭代法在什么情况下适用？

 

# 4       Torch和pandas的基本操作
Tensor转置：
​​torch.transpose(dim0, dim1)​​: 交换张量的两个指定维度。

​​tensor.permute(dims)​​: 更通用的方法，通过指定新维度的顺序来重新排列所有维度。

Tensor升维/降维

torch.unsqueeze(dim)或 tensor.unsqueeze(dim)在指定的 dim位置插入一个大小为1的新维度。

torch.squeeze()或 tensor.squeeze(dim)来移除所有大小为1的维度，或移除指定的单个大小为1的维度。

维度合并：

torch.cat(tensors, dim=0)将​​多个张量​​沿着一个​​现有的维度​​ dim连接起来。

Tensor堆叠

torch.stack(tensors, dim=0)将​​多个相同形状的张量​​沿着一个​​新的维度​​ dim堆叠起来

Tensor形状改变

使用 tensor.view(*shape)或 torch.reshape(*shape)在保持元素总数不变的情况下改变张量的形状。

Torch中的乘法有哪些？分别有哪些区别？

 

# 5       基本机器学习理论
1、 是否了解K-means聚类算法？简单讲述下K-means算法的步骤，K-means有哪些限制和缺陷？基于K-means改进的算法有哪些？

2、 是否了解DBSCAN聚类算法？简单讲述DBSCAN下算法的步骤，DBSCAN算法有哪些限制和缺陷？基于DBSCAN改进的算法有哪些？

3、 是否了解SVM(支持向量机)算法，能否将一下该算法的具体步骤？SVM算法有哪些限制和缺陷？基于SVM改进的算法有哪些？

4、 是否了解PCA(主成分分析)算法，能否将一下该算法的具体步骤？PCA算法有哪些限制和缺陷？基于PCA改进的算法有哪些？

5、 是否了解t-SNE算法，能否将一下该算法的具体步骤？t-SNE算法有哪些限制和缺陷？基于t-SNE改进的算法有哪些？

 

# 6       神经网络理论
1、 在深度神经网络中，为什么会出现的梯度消失和梯度爆炸？有哪些办法可以避免梯度消失或者梯度爆炸？

2、 是否了解BatchNorm，如果输入为(b, c, h, w)，那么可训练参数是多少？

3、 BatchNorm和LayerNorm的区别，分别各自的适用场景。

BatchNorm (BN) 的主场：计算机视觉 (CV) ，LayerNorm (LN) 的主场：自然语言处理 (NLP) & 序列模型

4、 讲一下交叉熵公式。

5、 讲一下对比学习的原理。

6、 简单介绍下InfoNCE的原理，其中温度系数有什么用？

7、 写一下softmax的函数定义式

8、 写一下adam优化器的数学公式

9、 是否了解Kimi K2 模型使用的Muon优化器？

 

# 7       Transformer架构
1、 推导transformer layer的计算复杂度，随着seq_len的增加。

2、 是否了解稀疏注意力机制？具体讲讲有哪些稀疏注意力。

3、 Padding的mask操作具体如何实现？

4、自注意力机制的公式，为什么要除以sqrt(d_k)

5、 为什么不使用维度更深的单一注意力机制，而是采用多头注意力？多头注意力(MHA)机制带来哪些优势？

6、 Transformer为什么使用LayerNorm而非BatchNorm？

7、Transformer中的encoder和decoder有什么区别？

8、如何设计实验验证Pre-LayerNorm和Post-LayerNorm对深层Transformer训练稳定性的影响？ 

9、Transformer中的残差连接是怎么处理的？具体作用是什么？

 

# 8       LLM训练
1、 你是否了解scaling law，展开讲一下，后续的scaling law有什么哪些改进？

2、讲讲KV cache

3、 结合RoPE位置编码，讲一下一个优秀的位置编码需要满足哪些特性？

4、 你是否知道RoPE编码有哪些后续改进？

5、大模型在训练时出现OOM(out of Memory)问题应该怎么解决？

6、 是否了解大模型训练的重计算？

7、 你是否了解大模型训练中的梯度累加策略，它有什么优势？与缩小batch_size相比有什么区别？

8、 讲下MHA/MQA/GQA。

9、 你是否了解deepseek v3用到的MLA机制？

10、在训练7B LLM时，如果使用AdamW优化器，那么它需要的峰值显存是多少？

11、介绍下Lora的原理

12、 Lora中的r和alpha这两个参数代表什么意思？

13、 为什么现在主流生成式大模型都是Decoder模式？

14、  SFT微调的loss是怎么计算的？

15、 如果给你一个千卡集群，去对qwen2.5-72B进行sft全参微调，在训练过程中可能会遇到什么困难？有哪些因素会导致模型训练失败，有什么应对方式？

16、如何确保数据pipeline在千卡训练中能够及时供给数据？预先应该做哪些处理？

17、LLM如果在训练过程中loss值出现spike应该怎么办？

18、 大模型训练时的并行方式有哪些？DP/PP/TP/CP具体有哪些区别？

19、如何减少分布式并行训练时出现的bubble 时间。

 

# 9       LLM推理
1、 vllm都有那些调优方法？

2、 如何在vllm推理时，保证大模型输出的确定性，有在vllm中哪些参数与之相关？

3、 是否了解LLM的分词器，LLM中的词表文件是如何生成的？

4、 vllm的swap和recompute是什么？

5、 vllm使用TP功能时，MHA的head数量不是GPU的整数倍怎么办？

6、 vllm的PageAttention具体是怎么实现的？里面的block大概是什么样的，会存储哪些信息？

7、 Flash Attention的online softmax怎么做的？难点在哪里？

8、 PD分离是什么？一定要做吗？最开始提出PD分离的背景是什么？

 

# 10       强化学习
1、 介绍一下RLHF

2、 简单介绍下PPO的训练流程。

3、 在LLM的PPO训练中，critic model和reward model的区别是什么？

4、 PPO中的广义优势函数是什么？为什么这么定义，其数学原理是什么？

5、 PPO中奖励稀疏，导致策略更新困难，应该怎么解决？

6、 简单介绍下DPO的训练流程。

7、 为什么DPO在训练LLM时，绕开了reward model？

8、 你在训练DPO时遇到了什么问题？是如何解决的？

9、 DPO有什么缺点？后续对DPO算法有哪些改进？

10、 DPO如果在训练中不稳定应该怎么处理？

11、 简单介绍下GRPO的训练流程。

12、 在使用GRPO训练LLM时，训练数据有什么要求？

13、 讲解一下GRPO的数据回放。

14、 强化学习中on-policy和off-policy的区别是什么？

15、 你在训练GRPO和DPO的时候使用了什么强化学习框架？

 

# 11       Agent相关
1、 是否了解MCP，简单讲一下MCP的工作流程

2、 举例一个你用到过的MCP的场景

3、 MCP有哪些缺点？

4、 是否使用过OpenAI Plugin、LangChain Agents相关的框架？

5、 如何让LLM Agent具有长期记忆能力(超过当前模型上下文窗口)

6、 你有什么办法避免LLM输出时的幻觉问题？

 

# 12      VLM相关模型
1、 ViT一般怎么进行预训练？

2、 是否了解 OpenAI 提出的Clip，它和SigLip有什么区别？为什么SigLip效果更好？

3、 Qwen2.5-VL的预训练流程。

4、 在Qwen2.5-VL中，一张560*560 pixel的图片，应该换算成多少个token？

 

# 13       GPU与CUDA算子
1、 CUDA出现bank conflict，应该怎么解决？

2、 cuda core的数量 与 开发算子中实际使用的线程 关系是什么？过量线程会发生什么情况？

3、GPU的内存结构是什么样的？

4、half2，float4这种优化 与 pack优化的底层原理是什么？

5、合并访存是什么？原理是什么？
