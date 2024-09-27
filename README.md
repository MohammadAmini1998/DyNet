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

### The architecture of each actor

- Weight generator: This block generates a weight based on the agent' observations.
- Message encoder: This block encodes messages from agent's observations
- Action selector: This block selects an action based on the agent's observations and message comming from other agetns. 
<p align="center">
  <img src="https://github.com/user-attachments/assets/aa062796-f864-4b4a-a669-a84e062678f9" alt="Capture" width="50%" />
</p>


## How to run the code

```bash
git clone [https://github.com/MohammadAmini1998/DyNet.git]
cd DyNet
python main.py
```
## Environment: Predator and Prey (PP)

- _n_ agents try to capture a randomly moving prey
- Observation: Position of themselves, relative positions of prey (heterogeneous observation range)
- Action: Move up/down/left/right
- Reward: Get reward when they capture the prey
- Performance metric: Number of steps taken to capture the prey

<img src="https://github.com/user-attachments/assets/9a788b87-d33f-49d3-af96-8654494008cb" width="500"/>

### Citation
This environment description is adapted from the paper:  
[Learning to Schedule Communication in Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1902.01554).


## Environment: Vehicular Networks

We consider an environment with \( M \) V2I links and \( K \) V2V links. The V2I links are each assigned fixed orthogonal spectrum sub-bands, where the \( m \)-th V2I link occupies the \( m \)-th sub-band, and all V2I transmission power remains constant. In contrast, the transmission power of the V2V links is dynamic and can vary. 

The primary challenge is to design an efficient spectrum-sharing scheme for the V2V links that maximizes both the sum capacity of the V2I links and the packet delivery rate of the V2V links.
### Citation 
L. Liang, H. Ye, and G. Y. Li, "Spectrum sharing in vehicular networks based on multi-agent reinforcement learning," IEEE Journal on Selected Areas in Communications, vol. 37, no. 10, pp. 2282-2292, Oct. 2019.


