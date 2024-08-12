# TeaMs-RL

See homepage: https://sites.google.com/view/teams-rl/home



 <div align=center>
 <img src="https://github.com/SafeRL-Lab/TeaMs-RL/blob/main/code/figures/media.png" width="850"/> 
 </div>
 <div style="text-align:justify;"> Figure 1: Comparative overview of LLM alignment techniques. Current methods (red shaded region) typically involve a two-phase process, starting with Supervised Fine-Tuning (SFT) of a pre-aligned LLM using a dataset of  human-crafted instructions and corresponding responses (often sourced from an expert LLM like ChatGPT), leading to a post-SFT LLM. This is then fine-tuned using RLHF, where human feedback on preferences is incorporated, resulting in a post-RLHF LLM. In contrast, our TeaMs-RL method (green shaded region) employs a single-phase SFT approach, initially utilizing RL for teaching expert LLMs to generate high-quality instructions. We train an RL policy (instructor LLM) to create diverse instructions (with the diversity evaluated by a reviewer LLM as a reward signal). Once trained, the instructor LLM produces a set of actions to teach an expert LLM to generate high-quality instructions, and the instructions are leveraged to query the expert LLM to form the SFT instruction dataset. This approach capitalizes on the strengths of RL to enhance the complexity of instructions and consequently the value of responses from the expert LLM.</div>


## Examples
 <div align=center>
 <img src="https://github.com/SafeRL-Lab/TeaMs-RL/blob/main/code/figures/teams-rl-example-1.png" width="550"/> 
   <img src="https://github.com/SafeRL-Lab/TeaMs-RL/blob/main/code/figures/teams-rl-example-1-02.png" width="550"/> 
 </div>
 <div style="center">  <center>Figure 2: Comparison Experiments of Solving General Tasks</center></div>


## Publication
If you find the repository useful, please cite the [paper](https://arxiv.org/pdf/2403.08694):
```
@article{gu2024teams,
  title={TeaMs-RL: Teaching LLMs to Teach Themselves Better Instructions via Reinforcement Learning},
  author={Gu, Shangding and Knoll, Alois and Jin, Ming},
  journal={arXiv preprint arXiv:2403.08694},
  year={2024}
}
```
