# distributed_of_vit
Vit分布式训练
Vision Transformer (ViT) 是一种非常流行的图像分类算法，其在使用自注意力机制处理图像的过程中取得了非常好的效果。然而，对于大规模的数据集和模型，单机训练的效率和速度都无法满足需求，因此需要使用分布式训练的方法。

分布式训练的基本原理
分布式训练是指将一个大规模的训练任务分解成若干个小任务，在多个计算节点上并行地执行这些任务，并通过交换参数和梯度信息来实现全局模型的训练。在分布式训练中，常用的两种架构是数据并行和模型并行。

在数据并行的架构中，每个计算节点都拥有一份完整的模型，但是只有一部分数据。每个节点使用自己的数据来训练模型，并将训练得到的参数和梯度发送给其他节点，最终通过聚合各节点的参数和梯度来更新全局模型。

在模型并行的架构中，每个计算节点只拥有模型的一部分。每个节点使用自己的部分模型来处理数据，并将处理结果发送给其他节点，最终通过组合各节点的模型来得到全局模型。

Vit分布式训练的实现
对于ViT模型的分布式训练，我们可以使用数据并行的架构。具体来说，我们将整个数据集分成若干个小批次，每个计算节点负责处理一部分批次的数据。每个节点使用自己的数据来训练模型，并将训练得到的参数和梯度发送给其他节点。最终，我们通过梯度聚合的方法来更新全局模型。

>对应数据需要预先运行pre_trainer.py进行预训练fine-tuning，在运行main.py。
>如果用其他数据，请修改dataset.py里面的函数。<br>
  ```torchrun --standalone -nproc-per-node=gpu pre_trainer.py```<br>
  ```python main.py```
