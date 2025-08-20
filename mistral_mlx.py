from mlx_lm import load, generate
model, tokenizer = load("mlx-community/SmolLM2-1.7B-Instruct")
text = generate(model, tokenizer, prompt="こんにちは、調子はどう？", verbose=True)
print(text)
