import 'dotenv/config';

import { parallelizationAgent } from 'agents/parallelization.agent';
import colors from 'colors';
import { v4 as uuid } from 'uuid';

async function main() {
    const threadId = uuid();
    const review =
        'The product itself is quite good, especially the build quality. However, the customer service was incredibly slow to respond, and the price feels a bit high for what you get. You should really consider offering faster support channels and maybe look into a more competitive pricing strategy.';
    const response = await parallelizationAgent.invoke(threadId, review);
    console.log(colors.blue(JSON.stringify(response, null, 2))); // Convert object to pretty-printed JSON string
}

main().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
