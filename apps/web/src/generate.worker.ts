import { generate as wasmGenerate } from "./holo";

self.onmessage = async (e) => {
  const { holoBytes, prompt, genOpts, tokenizerBytes } = e.data;
  
  try {
    const result = await wasmGenerate(
      holoBytes, 
      prompt, 
      genOpts, 
      tokenizerBytes, 
      (text: string) => {
        self.postMessage({ type: 'token', text });
      }
    );
    self.postMessage({ type: 'done', text: result });
  } catch (err: any) {
    self.postMessage({ type: 'error', error: err.toString() });
  }
};
