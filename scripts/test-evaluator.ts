import colors from 'colors';
import 'dotenv/config';
import { v4 as uuid } from 'uuid';

import { evaluatorOptimizerAgent } from 'agents/evaluator-optimizer.agent';

async function main() {
    const threadId = uuid();
    const productInfo = "Our new AI-powered noise-cancelling headphones 'AuraSilence Pro'";
    const audience = 'Remote workers and frequent travelers';
    const tone = 'Professional but exciting';
    const maxIterations = 3; // Limit iterations for the example

    console.log(colors.blue(`Starting subject line generation for: "${productInfo}" with Thread ID: ${threadId}`));

    const result = await evaluatorOptimizerAgent.invoke(threadId, productInfo, audience, tone, maxIterations);
    console.log('\n--- Final Result ---'.yellow);
    if (result.finalSubjectLine) {
        console.log('Accepted Subject Line:'.green, result.finalSubjectLine);
        console.log('Final Feedback:'.blue, result.feedback);
        console.log('Iterations:'.magenta, result.iterations);
    } else {
        console.error('Agent failed or max iterations reached:'.red, result.error);
        console.log('Last Attempt:'.red, result.lastSubjectLine);
        console.log('Last Feedback:'.red, result.lastFeedback);
        console.log('Iterations:'.red, result.iterations);
    }
}

main().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
