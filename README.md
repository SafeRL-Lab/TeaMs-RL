
<div align="center">
 
  
<h1 align="center" style="font-size: 30px;"><strong><em>TeaMs-RL</em></strong>: Teaching LLMs to Generate Better Instruction Datasets via Reinforcement Learning</h1>
<p align="center">
    <a href="https://arxiv.org/pdf/2403.08694">Paper</a>
    ·
    <a href="https://sites.google.com/view/teams-rl/home">Website</a>
    ·
    <a href="https://github.com/SafeRL-Lab/TeaMs-RL">Code</a>
    ·
    <a href="https://github.com/SafeRL-Lab/TeaMs-RL/issues">Issue</a>
  </p>
   <a href="https://github.com/SafeRL-Lab/TeaMs-RL/tree/main">
    <img src="https://github.com/SafeRL-Lab/TeaMs-RL/blob/main/Teams_RL_GPT/docs/overview-framework.png" alt="Logo" width="880">
  </a>
    
  <p align="left" style="font-size: 25px;"><strong>Comparative overview of LLM alignment techniques.</strong> Current methods (red shaded region) typically involve a two-phase process, starting with Supervised Fine-Tuning (SFT) of a pre-aligned LLM using a dataset of human-crafted instructions and corresponding responses (often sourced from an expert LLM like ChatGPT), leading to a post-SFT LLM. This is then fine-tuned using RLHF, where human feedback on preferences is incorporated, resulting in a post-RLHF LLM. In contrast, our TeaMs-RL method (green shaded region) employs a single-phase SFT approach, initially utilizing RL for teaching expert LLMs to generate high-quality instructions. We train an RL policy (instructor LLM) to create diverse instructions (with the diversity evaluated by a reviewer LLM as a reward signal). Once trained, the instructor LLM produces a set of actions to teach an expert LLM to generate high-quality instructions, and the instructions are leveraged to query the expert LLM to form the SFT instruction dataset. This approach capitalizes on the strengths of RL to enhance the complexity of instructions and consequently the value of responses from the expert LLM. Note: The expert LLMs are not involved in the training of the RL policy; we only use the instructor and the reviewer LLM for training the RL policy. The expert LLM is used to generate instructions and corresponding responses under the guidance of the trained RL policy.</p>
  <a href="https://github.com/SafeRL-Lab/TeaMs-RL/tree/main">
    <img src="https://github.com/SafeRL-Lab/TeaMs-RL/blob/main/Teams_RL_GPT/docs/RL-search.png" alt="Logo" width="580">
  </a>  
  <p align="left" style="font-size: 25px;"><strong>RL policy search for LLM instruction generation.</strong> A denotes actions (prompts), S denotes states (instructions), and The green line indicates the optimal instruction generation achieved during the policy search.</p>
</div>





# Run experiments



Install:


`pip install -r requirements.txt`

`pip install -e .`

run experiments

cd Teams_RL_GPT/teams_rl/runner/

`sh run_llm_rl.sh`


# An example

To clarify our methodology and facilitate replication, we provide a detailed example illustrating the inputs and outputs at each stage of our process, similar to Figure 1 in our [paper](https://arxiv.org/pdf/2403.08694).

---

### **Stage 1: Policy Training**

**Components:**
- **RL Policy:** (derived from Instructor Model) Learns to select actions based on the reviewer's feedback to improve the instructions.
- **Reviewer Model:** Provides feedback on the instructions.

**Example:**

1. **RL Policy Training**
With any Initial Instructions such as:

   ```
   "Describe the process of photosynthesis."
   ```
   The RL policy with Instructor Models learns to select actions that diversify instructions based on feedback from the reviewer model.

2. **RL Policy Training:**

   Based on the reviewer's feedback, the RL policy learns to select actions such as:
     - **Add Constraints**
     - **Deep Reasoning**
     - **Width Reasoning**

---

### **Stage 2: RL Policy Action Selection**

**Selected Action:** "Add Constraints"

**Purpose:** To make the instruction more challenging and comprehensive by adding specific constraints or requirements.

---

### **Stage 3: Guiding Expert LLMs**
**Purpose:** To make the instruction more challenging and comprehensive by adding specific constraints or requirements.

RL Policy Action Selection such as: **"Add Constraints"**

The RL policy generates a specialized prompt based on this action to guide the expert LLM in rewriting the instruction.

**Add Constraints: Action Prompt to Expert LLM:**

```
I want you to act as a Prompt Rewriter.

Your objective is to rewrite the given prompt into a more complex version to make it more challenging for AI systems like ChatGPT and GPT-4.

Ensure that the rewritten prompt remains reasonable, understandable, and suitable for human response.

Do not omit any non-text parts such as tables or code in the given prompt.

Do not repeat conditions or requirements in your response, and do not disclose your role.

Provide only the rewritten prompt without any introduction or explanation.

The new prompt should not exceed 2048 words.

You should complicate the given prompt by adding one more constraint or requirement.

Try not to make the rewritten prompt verbose; you can only add or replace 10 to 20 words in the given prompt.

Do not include phrases like 'Given Prompt' or 'Rewritten Prompt' in your response.

Given Prompt:
"Describe the process of photosynthesis."
```

---

### **Stage 4: Expert LLM Generates Instruction**

**Expert LLM Output:**

```
"Describe the process of photosynthesis in plants and explain how it varies in different environmental conditions."
```

---

### **Explanation:**

- **Original Instruction:** Simple and straightforward.
- **Rewritten Instruction:** Adds the constraint of explaining variations in different environmental conditions, increasing complexity and depth.
- **Compliance with Guidelines:** The expert LLM added approximately 12 words, adhering to the limit of adding or replacing 10 to 20 words.

---

Once we have the generation instructions, we use them to query the expert model for corresponding responses. After obtaining the responses, we have the final dataset (instructions and corresponding responses), which is then used to fine-tune foundation models.

### **Summary:**

This example demonstrates how our RL policy guides expert LLMs to generate more complex and high-quality instructions by selecting appropriate actions. The process ensures that the generated instructions are challenging yet reasonable, facilitating the creation of a valuable dataset for training advanced AI models.

---


<!-- <div align=center>
 <img src="https://github.com/SafeRL-Lab/TeaMs-RL/blob/main/code/figures/media.png" width="850"/> 
 </div>
 <div style="text-align:justify;"> Figure 1: Comparative overview of LLM alignment techniques. Current methods (red shaded region) typically involve a two-phase process, starting with Supervised Fine-Tuning (SFT) of a pre-aligned LLM using a dataset of  human-crafted instructions and corresponding responses (often sourced from an expert LLM like ChatGPT), leading to a post-SFT LLM. This is then fine-tuned using RLHF, where human feedback on preferences is incorporated, resulting in a post-RLHF LLM. In contrast, our TeaMs-RL method (green shaded region) employs a single-phase SFT approach, initially utilizing RL for teaching expert LLMs to generate high-quality instructions. We train an RL policy (instructor LLM) to create diverse instructions (with the diversity evaluated by a reviewer LLM as a reward signal). Once trained, the instructor LLM produces a set of actions to teach an expert LLM to generate high-quality instructions, and the instructions are leveraged to query the expert LLM to form the SFT instruction dataset. This approach capitalizes on the strengths of RL to enhance the complexity of instructions and consequently the value of responses from the expert LLM.</div>
 


## Examples
 <div align=center>
 <img src="https://github.com/SafeRL-Lab/TeaMs-RL/blob/main/code/figures/teams-rl-example-1.png" width="550"/> 
   <img src="https://github.com/SafeRL-Lab/TeaMs-RL/blob/main/code/figures/teams-rl-example-1-02.png" width="550"/> 
 </div>
 <div style="center">  <center>Figure 2: Comparison Experiments of Solving General Tasks</center></div>
-->

## Publication
If you find the repository useful, please cite the [paper](https://arxiv.org/pdf/2403.08694):
```
@article{gu2024teams,
  title={TeaMs-RL: Teaching LLMs to Generate Better Instruction Datasets via Reinforcement Learning},
  author={Gu, Shangding and Knoll, Alois and Jin, Ming},
  journal={arXiv preprint arXiv:2403.08694},
  year={2024}
}
```
## Acknowledgments

We thank the contributors from [llama-x](https://github.com/AetherCortex/Llama-X).
