import { MacroCausalOrchestrator } from './orchestrators/MacroCausalOrchestrator';
import { OpenAIService } from './services/OpenAIService';
import { ModelService } from './services/ModelService';

// Test configuration
const TEST_OPENAI_API_KEY = process.env.TEST_OPENAI_API_KEY || 'test-key';
const TEST_MODEL_SERVING_URL = process.env.TEST_MODEL_SERVING_URL || 'http://localhost:8000';

async function testSystem() {
  console.log('🧪 Testing Macro-Causal AI Agents System...\n');

  try {
    // Initialize services
    console.log('1. Initializing services...');
    const openAIService = new OpenAIService(TEST_OPENAI_API_KEY);
    const modelService = new ModelService(TEST_MODEL_SERVING_URL);
    
    console.log('✅ Services initialized successfully');

    // Initialize orchestrator
    console.log('\n2. Initializing orchestrator...');
    const orchestrator = new MacroCausalOrchestrator(openAIService, modelService);
    console.log('✅ Orchestrator initialized successfully');

    // Test queries
    const testQueries = [
      "What is the causal effect of monetary policy tightening on GDP growth?",
      "How do market regimes affect investment returns?",
      "What is the weather like today?", // This should be out of scope
      "Analyze the treatment effect of fiscal stimulus on unemployment"
    ];

    console.log('\n3. Testing queries...\n');

    for (let i = 0; i < testQueries.length; i++) {
      const query = testQueries[i];
      console.log(`--- Test Query ${i + 1} ---`);
      console.log(`Query: "${query}"`);
      
      try {
        const startTime = Date.now();
        const result = await orchestrator.executeQuery(query);
        const executionTime = Date.now() - startTime;
        
        console.log(`✅ Success: ${result.success}`);
        console.log(`⏱️  Execution Time: ${executionTime}ms`);
        
        if (result.success) {
          console.log(`📊 Analysis Type: ${result.metadata?.analysisType || 'unknown'}`);
          console.log(`🔍 Complexity: ${result.metadata?.complexity || 'unknown'}`);
          console.log(`📈 Steps Completed: ${result.metadata?.stepsCompleted || 0}/${result.metadata?.totalSteps || 0}`);
          
          if (result.modelResults) {
            const modelCount = Object.keys(result.modelResults).length;
            console.log(`🤖 Models Executed: ${modelCount}`);
          }
          
          console.log(`💬 Response Length: ${result.response?.length || 0} characters`);
        } else {
          console.log(`❌ Out of Scope: ${result.outOfScopeReason || 'Unknown reason'}`);
        }
        
      } catch (error) {
        console.log(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
      
      console.log(''); // Empty line for readability
    }

    console.log('🎉 System test completed!');

  } catch (error) {
    console.error('💥 System test failed:', error);
    process.exit(1);
  }
}

// Run the test if this file is executed directly
if (require.main === module) {
  testSystem().catch(console.error);
}

export { testSystem };
