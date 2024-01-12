from vllm import LLM, SamplingParams

class InferlessPythonModel:
    def initialize(self):

        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95,max_tokens=256)
        self.llm = LLM(model="TheBloke/dolphin-2.5-mixtral-8x7b-GPTQ", quantization="gptq", dtype="float16")

    def infer(self, inputs):
        prompts = inputs["prompt"]
        result = self.llm.generate(prompts, self.sampling_params)
        result_output = [output.outputs[0].text for output in result]
        print(result_output)
        return {'generated_result': result_output[0]}

    def finalize(self):
        pass