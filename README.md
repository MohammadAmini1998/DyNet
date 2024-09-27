# Dynamic Bandwidth Allocation in Cooperative Multi-Agent Reinforcement Learning

This is the code for implementing the __DyNet__ algorithm presented in the paper which will appear :: ["Dynamic Bandwidth Allocation in Cooperative
Multi-Agent Reinforcement Learning"]().

## Summary of the paper 
Current Multi-Agent Reinforcement Learning
(MARL) algorithms typically necessitate that agents transmit
a fixed amount of information at each time step, which
often results in sub-optimal bandwidth utilization. Agents are
frequently constrained to a binary decision: either transmit a
predetermined number of messages or none. This can lead to
inefficiencies, particularly when multiple agents are using the
same communication channel, thereby causing bottlenecks and
degrading overall performance. To address this limitation, we
propose a novel MARL algorithm named Dynamic Network
(DyNet), an Actor-Critic framework that dynamically allocates
bandwidth to each agent based on its needs, utilizing a Deep
Q-Network (DQN) for bandwidth allocation. Furthermore, we
introduce a penalty term within the objective function, which
penalizes communication based on the number of messages
transmitted. This approach ensures that agents communicate
only when necessary. We validate the practical application of this
algorithm in vehicular networks, demonstrating its effectiveness in real-world scenarios

### The overall architecture of DyNet

- Actor: Cinsists of action selector, message encoder and weight generator
- Scheduler: Map from weights __w__ to badnwidth __b__
- Critic: gives value and action values to the actor

<p align="center">
  <img src="https://github.com/user-attachments/assets/f8f39905-f39c-4df7-9623-393fe7c2c0aa" alt="Capture" width="50%" />
</p>
