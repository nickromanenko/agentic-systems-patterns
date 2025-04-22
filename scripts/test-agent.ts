import 'dotenv/config';
import { v4 as uuid } from 'uuid';

import { autonomousAgent } from 'agents/autonomous.agent';

async function main() {
    const threadId = uuid();
    const researchTopic = 'Tell me about LangGraph agents.';
    const maxIterations = 3; // Limit iterations for the example

    console.log(`Starting research for: "${researchTopic}" with Thread ID: ${threadId}`);

    const result = await autonomousAgent.invoke(threadId, researchTopic, maxIterations);
    console.log('\n--- Final Result ---');
    if (result.finalAnswer) {
        console.log('Final Answer:', result.finalAnswer);
    } else {
        console.error('Agent failed or max iterations reached:', result.error);
        console.log('Last Log:', result.lastLog); // If max iterations reached
    }
    console.log('Iterations:', result.iterations);
    console.log('Intermediate Steps:', result.intermediateSteps);
}

main().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
