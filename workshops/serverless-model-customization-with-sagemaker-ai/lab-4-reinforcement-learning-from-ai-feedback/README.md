# Lab X: Reinforcement Learning from AI Feedback (RLAIF)

In this lab, you will fine-tune a foundation model, in this case [Qwen 2.5 - 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), using a technique called **Reinforcement Learning from AI Feedback (RLAIF)**.

RLAIF substitutes human annotation in RLHF with an AI evaluator. A language model, guided by carefully designed prompts specifying evaluation criteria, serves directly as the reward model rather than being trained from human rankings.

## How RLAIF works in SageMaker Serverless Customization

1. **Generate responses**: For each training prompt, the model generates one or more candidate responses

2. **AI judge evaluation**: Each response is evaluated by a separate AI model (the "judge"). In this lab, we use [GPT OSS 120B](https://huggingface.co/openai/gpt-oss-120b) via Amazon Bedrock (`bedrock/openai.gpt-oss-120b-1:0`). The judge uses a carefully crafted prompt template that defines the evaluation criteria for the task. Based on these criteria, the judge assigns each response a numerical score

3. **Compute advantages**: The scores are used to calculate advantages, which indicate how good each response is relative to others. In this lab, we use GRPO (Group Relative Policy Optimization), which compares responses within each group to compute relative advantages from the AI judge's scores

4. **Update the model**: The advantages are used to update the model's parameters through reinforcement learning, increasing the probability of generating higher-scoring responses and decreasing the probability of lower-scoring ones

## Further Reading

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - Anthropic's early work on using AI feedback for alignment
- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267) - Google's systematic study of RLAIF