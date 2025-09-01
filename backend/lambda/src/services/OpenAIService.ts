import { ChatOpenAI } from "@langchain/openai";

export class OpenAIService {
  private model: ChatOpenAI;
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
    this.model = new ChatOpenAI({
      openAIApiKey: this.apiKey,
      modelName: "gpt-4",
      temperature: 0.1,
      maxTokens: 2000
    });
  }

  getModel(): ChatOpenAI {
    return this.model;
  }

  async testConnection(): Promise<boolean> {
    try {
      const response = await this.model.invoke("Hello, this is a test message.");
      return !!response.content;
    } catch (error) {
      console.error('OpenAIService.testConnection - Error:', error);
      return false;
    }
  }

  async validateApiKey(): Promise<boolean> {
    try {
      // Make a simple request to validate the API key
      const response = await this.model.invoke("Test");
      return response.content.length > 0;
    } catch (error) {
      console.error('OpenAIService.validateApiKey - Error:', error);
      return false;
    }
  }

  getModelInfo() {
    return {
      modelName: "gpt-4",
      temperature: 0.1,
      maxTokens: 2000,
      apiKeyConfigured: !!this.apiKey
    };
  }
}
