import 'dotenv/config';

import { orchestratorWorkerAgent } from 'agents/orchestrator-workers.agent';
import colors from 'colors';
import { v4 as uuid } from 'uuid';

async function main() {
    const threadId = uuid();
    const topic = 'Explain the main concepts behind Large Language Models (LLMs)';
    console.log(colors.blue(`Starting research on: "${topic}" with Thread ID: ${threadId}`));
    const result = await orchestratorWorkerAgent.invoke(threadId, topic);
    console.log('\n--- Final Result ---'.yellow);
    if (result.final_report) {
        console.log(result.final_report);
    } else {
        console.error('Agent failed:'.red, result.error);
    }

    // You could potentially retrieve the full state history using the checkpointer
    // const finalFullState = await orchestratorWorkerAgent.getState(threadId);
    // console.log('\n--- Final State ---');
    // console.log(JSON.stringify(finalFullState, null, 2));
}

main().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
