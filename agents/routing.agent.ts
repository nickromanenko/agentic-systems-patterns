/**
 * Use Case: Customer Service Query Routing
 *
 * This agent demonstrates the Routing pattern for handling customer service queries.
 * 1. Route Query: Takes a customer query and classifies it into predefined categories (e.g., Billing, Technical Support, General Inquiry).
 * 2. Handle Specific Category: Based on the classification, the query is routed to a specialized handler (LLM call with a specific prompt) for that category.
 * 3. Generate Response: The specialized handler generates an appropriate response.
 *
 * This allows for tailored responses and potentially different logic or tools for each query type,
 * improving efficiency and accuracy compared to a single monolithic prompt.
 */

import { AIMessage, BaseMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'; // Add ToolMessage
import { JsonOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';
import { RunnableConfig } from '@langchain/core/runnables';
import { tool } from '@langchain/core/tools';
import { CompiledStateGraph, END, MemorySaver, START, StateGraph, StateGraphArgs } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod'; // Import zod for schema definition

// Define the state schema for the graph
interface AgentState {
    query: string;
    category: 'billing' | 'tech_support' | 'general_inquiry' | 'unknown';
    response: string | null;
    messages: BaseMessage[]; // To keep track of the conversation history for the checkpointer
}

const agentState: StateGraphArgs<AgentState>['channels'] = {
    query: {
        value: (x?: string, y?: string) => y ?? x ?? '', // Always update query if provided
        default: () => '',
    },
    category: {
        value: (x: AgentState['category'], y: AgentState['category']) => y ?? x, // Update category
        default: () => 'unknown',
    },
    response: {
        value: (x: string | null, y: string | null) => y ?? x, // Update response
        default: () => null,
    },
    messages: {
        value: (x: BaseMessage[], y: BaseMessage[]) => x.concat(y), // Append messages
        default: () => [],
    },
};

// Define the router output schema
interface RouterOutput {
    category: AgentState['category'];
    reasoning: string;
}

class RoutingAgent {
    // Revert to simpler graph typing
    private graph: StateGraph<AgentState>;
    private compiledGraph: CompiledStateGraph<AgentState, Partial<AgentState>>;
    private checkpointer = new MemorySaver();
    private llm = new ChatOpenAI({ model: 'gpt-4o', temperature: 0 }); // Use a capable model for routing and responses

    constructor() {
        // Revert to simpler instantiation
        this.graph = new StateGraph({ channels: agentState });

        // Define Nodes using arrow functions assigned to properties and string literals
        this.graph.addNode('route_query', this.routeQuery);
        this.graph.addNode('handle_billing', this.handleBilling);
        this.graph.addNode('handle_tech_support', this.handleTechSupport);
        this.graph.addNode('handle_general_inquiry', this.handleGeneralInquiry);
        this.graph.addNode('handle_unknown', this.handleUnknown); // Node for unclassified queries

        // Define Edges using string literals and 'as any' for type errors
        this.graph.addEdge(START, 'route_query' as any); // Use 'as any'

        // Conditional Edge: Based on the classified category using string literals
        // The keys in the map ('billing', etc.) must match the return values of decideNextNode
        // Use 'as any' for the source node here as well
        this.graph.addConditionalEdges('route_query' as any, this.decideNextNode, {
            billing: 'handle_billing' as any, // Use 'as any'
            tech_support: 'handle_tech_support' as any, // Use 'as any'
            general_inquiry: 'handle_general_inquiry' as any, // Use 'as any'
            unknown: 'handle_unknown' as any, // Use 'as any'
        });

        // Edges from handlers to END using string literals
        this.graph.addEdge('handle_billing' as any, END); // Use 'as any'
        this.graph.addEdge('handle_tech_support' as any, END); // Use 'as any'
        this.graph.addEdge('handle_general_inquiry' as any, END); // Use 'as any'
        this.graph.addEdge('handle_unknown' as any, END); // Use 'as any'

        // Compile the graph
        this.compiledGraph = this.graph.compile({ checkpointer: this.checkpointer });
    }

    // --- Node Functions (as arrow function properties) ---

    private routeQuery = async (state: AgentState): Promise<Partial<AgentState>> => {
        console.log('--- Routing Query ---');
        const routingPrompt = PromptTemplate.fromTemplate(
            `You are an expert at routing customer service queries.
Classify the user's query into one of the following categories: "billing", "tech_support", or "general_inquiry".
Provide a brief reasoning for your classification.

User Query:
{query}

Respond with a JSON object containing "category" and "reasoning".
Example format: {{"category": "billing", "reasoning": "The user is asking about their recent invoice."}}`,
        );

        const parser = new JsonOutputParser<RouterOutput>();
        const routingChain = routingPrompt.pipe(this.llm).pipe(parser);

        let category: AgentState['category'] = 'unknown';
        let aiMessage: BaseMessage;

        try {
            const result = await routingChain.invoke({ query: state.query });
            console.log('Routing Result:', result);
            category = result.category;
            aiMessage = new AIMessage({ content: JSON.stringify(result) });
        } catch (error) {
            console.error('Routing failed:', error);
            aiMessage = new AIMessage({ content: `Routing failed: ${error}` });
            category = 'unknown'; // Default to unknown if parsing or LLM call fails
        }

        return { category: category, messages: [new HumanMessage('Classify my query'), aiMessage] };
    };

    private decideNextNode = (state: AgentState): AgentState['category'] => {
        console.log(`--- Deciding Next Node based on Category: ${state.category} ---`);
        return state.category;
    };

    private handleBilling = async (state: AgentState): Promise<Partial<AgentState>> => {
        console.log('--- Handling Billing Query ---');
        const prompt = ChatPromptTemplate.fromMessages([
            ['system', 'You are a billing support specialist. Provide a helpful and concise response regarding billing.'],
            ['human', '{query}'],
        ]);
        const chain = prompt.pipe(this.llm);
        const response = await chain.invoke({ query: state.query });
        console.log('Billing Response:', response.content);
        return { response: response.content as string, messages: [new HumanMessage('Handle billing'), response] };
    };

    // Define the tech support tool
    private techSupportTool = tool(
        async (input: { query: string }) => {
            const { query } = input;
            const lowerQuery = query.toLowerCase();

            // Common technical issues with hardcoded responses
            if (lowerQuery.includes('password') && (lowerQuery.includes('reset') || lowerQuery.includes('forgot'))) {
                return 'To reset your password:\n1. Go to our login page\n2. Click "Forgot Password"\n3. Enter your email\n4. Check your email for reset instructions\n5. Follow the link to create a new password';
            }

            if (lowerQuery.includes('login') && (lowerQuery.includes('problem') || lowerQuery.includes('issue'))) {
                return "For login issues:\n1. Ensure you're using the correct email/username\n2. Check your password is correct\n3. Try clearing browser cookies/cache\n4. Try a different browser\nIf still having issues, contact support@example.com";
            }

            if (lowerQuery.includes('slow') || lowerQuery.includes('performance')) {
                return 'For performance issues:\n1. Check your internet connection\n2. Close other tabs/applications\n3. Clear browser cache\n4. Try a different browser\n5. Restart your device';
            }

            return 'No specific solution found for this technical issue. Please provide more details about your problem.';
        },
        {
            name: 'tech_support_handler',
            description: 'Handles common technical support queries with predefined solutions.',
            schema: z.object({
                query: z.string().describe('The user query related to a technical issue.'),
            }),
        },
    );

    private handleTechSupport = async (state: AgentState): Promise<Partial<AgentState>> => {
        console.log('--- Handling Technical Support Query (LLM decides tool use) ---');

        // Bind the tool to the LLM for this specific node/chain
        const llmWithTool = this.llm.bindTools([this.techSupportTool]);

        // Prompt instructing the LLM about the tool
        const prompt = ChatPromptTemplate.fromMessages([
            [
                'system',
                `You are a technical support specialist. You have access to a tool called 'tech_support_handler' that can provide standard answers for common issues like password resets, login problems, and slow performance.
If the user's query clearly matches one of these common issues, use the 'tech_support_handler' tool.
Otherwise, provide a helpful troubleshooting response based on the query.`,
            ],
            ['human', '{query}'],
        ]);

        const chain = prompt.pipe(llmWithTool);
        const messages: BaseMessage[] = [new HumanMessage('Handle tech support (LLM decides tool use)')];
        let finalResponse: string;

        try {
            const llmResponse = await chain.invoke({ query: state.query });
            messages.push(llmResponse); // Add LLM response to history

            // Check if the LLM decided to call the tool
            const toolCalls = llmResponse.tool_calls ?? [];
            if (toolCalls.length > 0 && toolCalls[0].name === 'tech_support_handler') {
                console.log('LLM decided to use tech_support_handler tool.');
                const toolCall = toolCalls[0];
                // Invoke the tool with the arguments provided by the LLM
                const toolOutput = await this.techSupportTool.invoke(toolCall.args);
                console.log('Tech Support Tool Response:', toolOutput);
                // Add tool output to messages for context
                messages.push(new ToolMessage({ content: toolOutput, tool_call_id: toolCall.id! }));
                finalResponse = toolOutput; // Use the tool's output as the final response
            } else {
                console.log('LLM generated direct response.');
                finalResponse = llmResponse.content as string; // Use the LLM's direct response
            }
        } catch (error) {
            console.error('Error during tech support handling:', error);
            finalResponse = `Sorry, an error occurred while handling your tech support request: ${error instanceof Error ? error.message : String(error)}`;
            messages.push(new AIMessage(finalResponse)); // Add error message
        }

        console.log('Final Tech Support Response:', finalResponse);
        return { response: finalResponse, messages: messages };
    };

    private handleGeneralInquiry = async (state: AgentState): Promise<Partial<AgentState>> => {
        console.log('--- Handling General Inquiry ---');
        const prompt = ChatPromptTemplate.fromMessages([
            ['system', 'You are a general customer service representative. Provide a helpful response to the inquiry.'],
            ['human', '{query}'],
        ]);
        const chain = prompt.pipe(this.llm);
        const response = await chain.invoke({ query: state.query });
        console.log('General Inquiry Response:', response.content);
        return { response: response.content as string, messages: [new HumanMessage('Handle general inquiry'), response] };
    };

    private handleUnknown = async (state: AgentState): Promise<Partial<AgentState>> => {
        console.log('--- Handling Unknown Query ---');
        const response = "I'm sorry, I couldn't categorize your request. Could you please rephrase or provide more details?";
        console.log('Unknown Response:', response);
        return { response: response, messages: [new HumanMessage('Handle unknown category'), new AIMessage(response)] };
    };

    // --- Public Invocation Method ---

    async invoke(threadId: string, query: string): Promise<string> {
        const initialState: AgentState = {
            query: query,
            category: 'unknown',
            response: null,
            messages: [new HumanMessage(query)],
        };

        const config: RunnableConfig = { configurable: { thread_id: threadId } };
        let finalState: AgentState | undefined;

        try {
            // Or just invoke and get the final state
            finalState = (await this.compiledGraph.invoke(initialState, config)) as AgentState; // Explicit cast

            if (finalState?.response) {
                return `Category: ${finalState.category}\nResponse:\n${finalState.response}`;
            } else {
                return 'An unexpected error occurred, and no response was generated.';
            }
        } catch (error) {
            console.error('Error invoking graph:', error);
            return `An error occurred while processing your request: ${error instanceof Error ? error.message : String(error)}`;
        }
    }
}

export const routingAgent = new RoutingAgent();
