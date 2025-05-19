from transformers import pipeline

class NLPPromptHandler:
    def __init__(self, model_name="google/flan-t5-base"):
        self.pipe = pipeline("text2text-generation", model=model_name)

    def parse_intent(self, prompt):
        response = self.pipe(prompt, max_new_tokens=50)[0]["generated_text"]
        return response

if __name__ == "__main__":
    handler = NLPPromptHandler()
    prompt = "Optimize the kernel for shared memory and reduce global memory access"
    print("Parsed Intent:", handler.parse_intent(prompt))
    