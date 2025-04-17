import 'dotenv/config';

import { promptChainingAgent } from 'agents/prompt-chaining.agent';
import colors from 'colors';
import { v4 as uuid } from 'uuid';

async function main() {
    const threadId = uuid();
    const topic = 'Brave knight saves Kylie Jenner from a dragon';
    const response = await promptChainingAgent.invoke(threadId, topic);
    console.log(colors.blue(response));
}

main().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
