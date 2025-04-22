import { AgentAction, AgentFinish, AgentStep } from '@langchain/core/agents';
import { AIMessage, BaseMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { RunnableConfig } from '@langchain/core/runnables';
import { DynamicStructuredTool, tool } from '@langchain/core/tools';
import { convertToOpenAITool } from '@langchain/core/utils/function_calling';
import { CompiledStateGraph, END, MemorySaver, START, StateGraph, StateGraphArgs } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { tavily } from '@tavily/core';
import { z } from 'zod';
const tavilyClient = tavily({ apiKey: process.env.TAVILY_API_KEY });

/**
 * Use Case: Autonomous Research Assistant (Agents Pattern)
 *
 * This agent demonstrates the core "Agents" pattern. Given a research topic, it autonomously
 * decides whether to use a tool (like web search) to gather information or to formulate a final
 * answer based on the information gathered so far. It interacts with its "environment"
 * (the results of tool calls) to inform its next step.
 *
 * Workflow:
 * 1. Input: Takes a research topic (e.g., "What is LangGraph?").
 * 2. Agent Node: An LLM acts as the agent's brain. It receives the current state (input, past steps)
 *    and decides the next action:
 *    - Call a Tool: If more information is needed, it specifies which tool to use and with what input
 *      (e.g., use 'web_search' for "LangGraph features").
 *    - Finish: If it has enough information, it formulates the final answer.
 * 3. Tool Executor Node: If the agent decided to use a tool, this node executes it (e.g., performs
 *    the web search) and gets the result (environment feedback).
 * 4. Loop: The tool result is added to the state and fed back into the Agent Node for the next decision.
 * 5. Repeat Steps 2-4: The cycle of deciding, acting (tool call), and observing (tool result)
 *    continues until the agent decides to finish.
 * 6. Output: The final synthesized answer to the research topic.
 *
 * This pattern is useful for open-ended tasks where the sequence of steps isn't fixed and relies
 * on runtime information gathering and decision-making.
 */

// --- State Definition ---
interface AutonomousAgentState {
    input: string; // The initial user query
    agent_outcome: AgentAction | AgentFinish | null; // The decision made by the agent LLM
    intermediate_steps: AgentStep[]; // History of tool calls and their results
    iterations: number; // Current loop count
    maxIterations: number; // Maximum allowed loops
    messages: BaseMessage[]; // Conversation history for the checkpointer
}

// --- Search Tool Definition ---
const searchTool = tool(
    async (input: { query: string }) => {
        const { query } = input;
        console.log(`--- Web Search Tool Called with input: "${query}" ---`);
        const response = await tavilyClient.search(query, { topic: 'general', searchDepth: 'basic' });
        const results = response.results.map(result => `Title: ${result.title}, URL: ${result.url}, Content: ${result.content}`);
        return results.join('\n=====\n');
    },
    {
        name: 'web_search',
        description: 'Simulates searching the web for information on a given topic.',
        schema: z.object({
            query: z.string().describe('The search query to find information.'),
        }),
    },
);

// --- Graph Channels Definition ---
const agentStateChannels: StateGraphArgs<AutonomousAgentState>['channels'] = {
    input: { value: null, default: () => '' },
    agent_outcome: { value: null, default: () => null },
    intermediate_steps: {
        value: (x: AgentStep[], y: AgentStep[]) => x.concat(y),
        default: () => [],
    },
    iterations: {
        value: (x: number, y: number) => x + y,
        default: () => 0,
    },
    maxIterations: { value: null, default: () => 5 }, // Default max iterations
    messages: {
        value: (x: BaseMessage[], y: BaseMessage[]) => x.concat(y),
        default: () => [],
    },
};

class AutonomousAgent {
    private graph: StateGraph<AutonomousAgentState>;
    private compiledGraph: CompiledStateGraph<AutonomousAgentState, Partial<AutonomousAgentState>>;
    private checkpointer = new MemorySaver();
    private agentLlm = new ChatOpenAI({ model: 'gpt-4o', temperature: 0 });
    private tools: DynamicStructuredTool[];
    private agentLlmWithTools;

    constructor() {
        this.tools = [searchTool];
        // Bind tools to the LLM so it knows how/when to call them
        this.agentLlmWithTools = this.agentLlm.bind({
            tools: this.tools.map(convertToOpenAITool),
        });

        this.graph = new StateGraph({ channels: agentStateChannels });

        // Define Nodes
        this.graph.addNode('agent', this.agentNode);
        this.graph.addNode('execute_tool', this.executeToolNode);

        // Define Edges
        this.graph.addEdge(START, 'agent' as any); // Use 'as any' for type compatibility
        // Conditional Edge: Route based on agent's decision
        this.graph.addConditionalEdges('agent' as any, this.shouldContinue, {
            // Use 'as any'
            continue: 'execute_tool' as any, // If agent decided to use a tool
            end: END, // If agent decided to finish or max iterations reached
        });
        // Edge from tool execution back to the agent for the next decision
        this.graph.addEdge('execute_tool' as any, 'agent' as any); // Use 'as any'

        // Compile the graph
        this.compiledGraph = this.graph.compile({ checkpointer: this.checkpointer });
    }

    // --- Node Functions ---

    // Agent Node: Decides the next action (call tool or finish)
    private agentNode = async (state: AutonomousAgentState): Promise<Partial<AutonomousAgentState>> => {
        console.log(`--- Agent Node (Iteration ${state.iterations + 1}) ---`);
        const { input, intermediate_steps, messages } = state;

        // Prepare messages for the LLM, including history and tool results
        const currentMessages: BaseMessage[] = [new HumanMessage(input)];
        for (const step of intermediate_steps) {
            currentMessages.push(
                new AIMessage({
                    content: '',
                    // Ensure args is an object and handle missing toolCallId via casting
                    tool_calls: [
                        {
                            name: step.action.tool,
                            args: typeof step.action.toolInput === 'string' ? { input: step.action.toolInput } : step.action.toolInput,
                            id: (step.action as any).toolCallId ?? `tool_${Date.now()}`,
                        },
                    ],
                }),
            );
            currentMessages.push(
                new ToolMessage({ content: step.observation, tool_call_id: (step.action as any).toolCallId ?? `tool_${Date.now()}` }),
            ); // Cast action to any for toolCallId
        }

        // Invoke the LLM with tools
        const response = await this.agentLlmWithTools.invoke(currentMessages);
        console.log('Agent LLM Response:', response);

        let agent_outcome: AgentAction | AgentFinish;
        // Check if the LLM decided to call a tool
        if (response.tool_calls && response.tool_calls.length > 0) {
            const toolCall = response.tool_calls[0]; // Assuming one tool call per step for simplicity
            // Explicitly cast to AgentAction and provide fallback for toolCallId
            agent_outcome = {
                tool: toolCall.name,
                toolInput: toolCall.args,
                log: `Agent decided to use tool: ${toolCall.name} with input ${JSON.stringify(toolCall.args)}`,
                toolCallId: toolCall.id ?? `tool_${Date.now()}`,
            } as AgentAction;
            console.log(`Agent Action: Call tool ${(agent_outcome as AgentAction).tool}`); // Cast needed due to union type
        } else {
            // LLM decided to finish
            agent_outcome = {
                returnValues: { output: response.content },
                log: `Agent decided to finish. Final Answer: ${response.content}`,
            };
            console.log(`Agent Action: Finish`);
        }

        // Increment iteration count
        return { agent_outcome, messages: [response], iterations: 1 };
    };

    // Tool Executor Node: Runs the chosen tool
    private executeToolNode = async (state: AutonomousAgentState): Promise<Partial<AutonomousAgentState>> => {
        console.log('--- Execute Tool Node ---');
        const { agent_outcome } = state;

        if (!agent_outcome || !('tool' in agent_outcome)) {
            throw new Error('No tool action found in agent outcome.');
        }

        const action = agent_outcome as AgentAction;
        const tool = this.tools.find(t => t.name === action.tool);

        if (!tool) {
            throw new Error(`Tool "${action.tool}" not found.`);
        }

        try {
            const observation = await tool.call(action.toolInput, { callbacks: undefined }); // Pass undefined or actual callbacks if needed
            console.log(`Tool ${action.tool} Observation:`, observation);
            const step: AgentStep = { action, observation };
            // Ensure 'tool_call_id' has a fallback.
            const toolMessage = new ToolMessage({ content: observation, tool_call_id: (action as any).toolCallId ?? `tool_${Date.now()}` }); // Cast action to any for toolCallId
            return { intermediate_steps: [step], messages: [toolMessage] };
        } catch (error) {
            console.error(`Error executing tool ${action.tool}:`, error);
            const observation = `Error executing tool ${action.tool}: ${error instanceof Error ? error.message : String(error)}`;
            const step: AgentStep = { action, observation };
            // Ensure 'tool_call_id' has a fallback, even in error case.
            const toolMessage = new ToolMessage({ content: observation, tool_call_id: (action as any).toolCallId ?? `tool_${Date.now()}` }); // Cast action to any for toolCallId
            // Still return the step so the agent knows the tool failed
            return { intermediate_steps: [step], messages: [toolMessage] };
        }
    };

    // --- Conditional Edge Logic ---

    // Determines whether to continue the loop (tool call) or end (finish/max iterations)
    private shouldContinue = (state: AutonomousAgentState): 'continue' | 'end' => {
        console.log('--- Checking Condition ---');
        const { agent_outcome, iterations, maxIterations } = state;

        if (iterations >= maxIterations) {
            console.log(`Max iterations (${maxIterations}) reached. Forcing end.`);
            // Optionally modify the outcome to indicate forced termination
            return 'end';
        }

        if (agent_outcome && 'tool' in agent_outcome) {
            console.log('Agent requested tool use. Continuing.');
            return 'continue';
        } else {
            console.log('Agent decided to finish. Ending.');
            return 'end';
        }
    };

    // --- Public Invocation Method ---
    async invoke(threadId: string, input: string, maxIterations: number = 5): Promise<Record<string, any>> {
        const initialState: Partial<AutonomousAgentState> = {
            input,
            maxIterations,
            messages: [new HumanMessage(input)],
        };

        const config: RunnableConfig = { configurable: { thread_id: threadId } };
        let finalState: AutonomousAgentState | undefined;

        console.log(`--- Invoking Autonomous Agent Graph for Thread ${threadId} ---`);
        try {
            // Using invoke to get the final state
            finalState = (await this.compiledGraph.invoke(initialState, config)) as AutonomousAgentState;

            console.log(`--- Invocation Complete for Thread ${threadId} ---`);
            console.log('Final State:', finalState);

            if (finalState?.agent_outcome && 'returnValues' in finalState.agent_outcome) {
                return {
                    finalAnswer: finalState.agent_outcome.returnValues.output,
                    iterations: finalState.iterations,
                    intermediateSteps: finalState.intermediate_steps,
                };
            } else if (finalState?.iterations >= finalState?.maxIterations) {
                const lastStep = finalState?.intermediate_steps?.[finalState.intermediate_steps.length - 1];
                const lastLog = lastStep ? `Last action: ${lastStep.action.tool}, Observation: ${lastStep.observation}` : 'No steps taken.';
                return {
                    error: `Agent reached maximum iterations (${finalState.maxIterations}) without finishing.`,
                    iterations: finalState.iterations,
                    intermediateSteps: finalState.intermediate_steps,
                    lastLog: lastLog,
                };
            } else {
                const lastMessage = finalState?.messages?.[finalState.messages.length - 1]?.content ?? 'Unknown error state.';
                return {
                    error: `Agent stopped unexpectedly. Last known message: ${lastMessage}`,
                    iterations: finalState?.iterations ?? 0,
                    intermediateSteps: finalState?.intermediate_steps ?? [],
                };
            }
        } catch (error) {
            console.error(`--- Error Invoking Graph for Thread ${threadId}:`, error);
            return { error: `An error occurred during the process: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
}

export const autonomousAgent = new AutonomousAgent();
