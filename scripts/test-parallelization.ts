import 'dotenv/config';

import { parallelizationAgent } from 'agents/parallelization.agent';
import colors from 'colors';
import { v4 as uuid } from 'uuid';

async function main() {
    const threadId = uuid();
    const review = '';
    const response = await parallelizationAgent.invoke(threadId, review);
    console.log(colors.blue(response));
}

main().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
