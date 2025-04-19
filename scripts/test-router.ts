import 'dotenv/config';

import { routingAgent } from 'agents/routing.agent';
import colors from 'colors';
import { v4 as uuid } from 'uuid';

async function main() {
    const threadId = uuid();
    const query = 'I have an issue with sign in. How can I reset my password?';
    const response = await routingAgent.invoke(threadId, query);
    console.log(colors.blue(response));
}

main().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
