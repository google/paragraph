# ParaGraph

## Disclaimer

This is not an official Google product. This project was created by
[Michael Isaev](https://www.github.com/michael-isaev) and
[Nic McDonald](https://www.github.com/nicmcd) at Google.

## What is it?
ParaGraph is a *Para*llel *Graph* representation of parallel computing applications that can be executed in a system level simulator. ParaGraph is designed to be an interface between the parallel program source code, and a system level simulator that should "execute" the program on the model of a distributed system. You can think about ParaGraph as an IR (Intermediate Representation) that can be interfaced with various simulators as a backend, just similar to how LLVM IR or MLIR can be interfaced with backends that target various hardware. This approach allows us to introduce accurate application models to system level simulation frameworks, and model parallel computing applications execution on the future distributed systems.

## How it works
Paragraph extracts high level computation and communication nodes from the compiled program or an execution trace, performs topology-based communication lowering, and rewrites the graph in the special format suitable for graph execution in a system simulator. Currently, we are targeting [Tensorflow](https://github.com/tensorflow/tensorflowhttps://github.com/tensorflow/tensorflow) and [PyTorch](https://github.com/pytorch/pytorch) programs through [XLA compiler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla). MPI programms are planned to be supported in the future.

## ParaGraph origin
Originally, ParaGraph was a summer 2020 internship project at Google that aimed to extract communication traffic from Machine Learning applications written in [TensorFlow](https://github.com/tensorflow/tensorflowhttps://github.com/tensorflow/tensorflow), and simulate it in [SuperSim](https://github.com/ssnetsim/supersim) event driven network simulator.
