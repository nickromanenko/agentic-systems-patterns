import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages';
import { JsonOutputParser } from '@langchain/core/output_parsers';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableConfig } from '@langchain/core/runnables';
import { CompiledStateGraph, END, MemorySaver, START, StateGraph, StateGraphArgs } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';

/**
 * Use Case: Marketing Email Subject Line Generation & Refinement (Evaluator-Optimizer Pattern)
 *
 * This agent demonstrates the Evaluator-Optimizer pattern. Given details about a product/offer
 * and target audience, it iteratively generates and refines a marketing email subject line.
 * One LLM (Generator) creates a subject line, while another LLM (Evaluator) critiques it
 * based on defined criteria (e.g., conciseness, engagement, clarity, tone). The process
 * loops, incorporating feedback, until the Evaluator accepts the subject line.
 *
 * Workflow:
 * 1. Input: Takes product details, target audience, and desired tone (e.g., "New running shoes", "Fitness enthusiasts", "Urgent & exciting").
 * 2. Generator Node: An LLM generates an initial subject line based on the input.
 *    (e.g., "Check out our new running shoes!")
 * 3. Evaluator Node: Another LLM evaluates the generated subject line against criteria.
 *    - If Accepted: The workflow ends, returning the subject line.
 *    - If Rejected: The Evaluator provides specific feedback for improvement.
 *      (e.g., {"decision": "reject", "feedback": "Make it more exciting and add urgency. Mention the limited-time discount."})
 * 4. Loop: If rejected, the feedback is passed back to the Generator Node, which uses it
 *    along with the original input to generate a revised subject line.
 *    (e.g., Generator tries again: "⚡️ Limited Time: 25% Off New SpeedRunner Shoes!")
 * 5. Repeat Steps 3-4: The evaluation and refinement loop continues until the Evaluator accepts the subject line.
 * 6. Output: The final, accepted email subject line.
 *
 * This pattern is useful for tasks requiring iterative refinement against clear quality standards,
 * where an LLM can both generate content and provide structured feedback for improvement.
 */

// --- State Definition ---
interface EvaluatorOptimizerState {
    productInfo: string;
    audience: string;
    tone: string;
    currentSubjectLine: string | null; // The subject line generated in the current iteration
    feedback: string | null; // Feedback from the evaluator
    evaluationResult: 'accept' | 'reject' | null; // Decision from the evaluator
    iterations: number; // To prevent infinite loops
    maxIterations: number; // Max loops allowed
    messages: BaseMessage[]; // History for checkpointer
}

// --- Output Schemas ---
interface EvaluatorOutput {
    decision: 'accept' | 'reject';
    feedback: string; // Feedback is required even if accepted, can be "Looks good!"
}

// --- Graph Channels Definition ---
const agentStateChannels: StateGraphArgs<EvaluatorOptimizerState>['channels'] = {
    productInfo: { value: null, default: () => '' },
    audience: { value: null, default: () => '' },
    tone: { value: null, default: () => '' },
    currentSubjectLine: { value: null, default: () => null },
    feedback: { value: null, default: () => null },
    evaluationResult: { value: null, default: () => null },
    iterations: {
        value: (x: number, y: number) => x + y, // Reducer to increment
        default: () => 0,
    },
    maxIterations: { value: null, default: () => 5 }, // Default max iterations
    messages: {
        value: (x: BaseMessage[], y: BaseMessage[]) => x.concat(y),
        default: () => [],
    },
};

class EvaluatorOptimizerAgent {
    private graph: StateGraph<EvaluatorOptimizerState>;
    private compiledGraph: CompiledStateGraph<EvaluatorOptimizerState, Partial<EvaluatorOptimizerState>>;
    private checkpointer = new MemorySaver();
    // Use different models or prompts for generator and evaluator for better results potentially
    private generatorLlm = new ChatOpenAI({ model: 'gpt-4.1-nano', temperature: 0.7 });
    private evaluatorLlm = new ChatOpenAI({ model: 'gpt-4.1', temperature: 0 }); // Evaluator might need to be more deterministic

    constructor() {
        this.graph = new StateGraph({ channels: agentStateChannels });

        // Define Nodes
        this.graph.addNode('generator', this.generatorNode);
        this.graph.addNode('evaluator', this.evaluatorNode);

        // Define Edges
        this.graph.addEdge(START, 'generator' as any); // Use 'as any'
        this.graph.addEdge('generator' as any, 'evaluator' as any); // Use 'as any'
        // Conditional Edge: Loop or End based on evaluation
        this.graph.addConditionalEdges('evaluator' as any, this.shouldContinue, {
            // Use 'as any'
            continue: 'generator' as any, // Loop back if rejected and within iteration limit
            end: END, // End if accepted or max iterations reached
        });

        // Compile the graph
        this.compiledGraph = this.graph.compile({ checkpointer: this.checkpointer });
    }

    // --- Node Functions ---

    // Generator: Creates or refines the subject line
    private generatorNode = async (state: EvaluatorOptimizerState): Promise<Partial<EvaluatorOptimizerState>> => {
        console.log(`--- Generator Node (Iteration ${state.iterations + 1}) ---`);
        const { productInfo, audience, tone, feedback } = state;

        let promptText = `You are a creative marketing copywriter. Generate a compelling email subject line based on the following details:
Product/Offer: {productInfo}
Target Audience: {audience}
Desired Tone: {tone}`;

        if (feedback) {
            promptText += `\n\nImprove the previous attempt based on this feedback: {feedback}`;
            console.log('Incorporating feedback:', feedback);
        } else {
            promptText += `\n\nGenerate an initial subject line.`;
        }

        promptText += `\n\nSubject Line:`;
    };

    // Evaluator: Critiques the subject line and decides whether to accept or reject
    private evaluatorNode = async (state: EvaluatorOptimizerState): Promise<Partial<EvaluatorOptimizerState>> => {
        console.log('--- Evaluator Node ---');
        const { productInfo, audience, tone, currentSubjectLine } = state;

        if (!currentSubjectLine || currentSubjectLine.startsWith('Error')) {
            console.log('Skipping evaluation due to generation error.');
            // Force end if generator failed
            return {
                evaluationResult: 'reject',
                feedback: 'Generator failed, cannot evaluate.',
                messages: [new AIMessage('Evaluation skipped due to generator error.')],
            };
        }

        const prompt = PromptTemplate.fromTemplate(
            `You are a strict marketing campaign evaluator. Evaluate the following email subject line based on the product details, audience, and desired tone.

Product/Offer: {productInfo}
Target Audience: {audience}
Desired Tone: {tone}
Subject Line to Evaluate: "{currentSubjectLine}"

Evaluation Criteria:
1.  **Clarity:** Is the subject line clear and easy to understand?
2.  **Relevance:** Is it relevant to the product/offer and audience?
3.  **Engagement:** Is it intriguing, exciting, or likely to encourage opens?
4.  **Conciseness:** Is it short enough (ideally under 50 characters)?
5.  **Tone:** Does it match the desired tone?

Respond with a JSON object containing two keys:
- "decision": Either "accept" or "reject".
- "feedback": Provide specific, actionable feedback for improvement if rejected. If accepted, briefly state why it's good (e.g., "Concise and engaging").

Example Reject Format: {{"decision": "reject", "feedback": "Too generic. Make it more specific to the discount and add urgency."}}
Example Accept Format: {{"decision": "accept", "feedback": "Clear, concise, and creates urgency."}}

Evaluation:`,
        );
        const parser = new JsonOutputParser<EvaluatorOutput>();
        const chain = prompt.pipe(this.evaluatorLlm).pipe(parser);
    };

    // --- Conditional Edge Logic ---

    // Determines whether to continue the loop or end
    private shouldContinue = (state: EvaluatorOptimizerState): 'continue' | 'end' => {
        console.log('--- Checking Condition ---');
        const { evaluationResult, iterations, maxIterations } = state;
    };

    // --- Public Invocation Method ---

    async invoke(threadId: string, productInfo: string, audience: string, tone: string, maxIterations: number = 5): Promise<Record<string, any>> {
        const initialState: Partial<EvaluatorOptimizerState> = {
            productInfo,
            audience,
            tone,
            maxIterations, // Set max iterations from input
            messages: [new HumanMessage(`Generate a subject line for: ${productInfo}, Audience: ${audience}, Tone: ${tone}`)],
        };

        const config: RunnableConfig = { configurable: { thread_id: threadId } };
        let finalState: EvaluatorOptimizerState | undefined;

        console.log(`--- Invoking Evaluator-Optimizer Graph for Thread ${threadId} ---`);
        try {
            // Stream events to see the flow (optional, but helpful for debugging)
            // const stream = this.compiledGraph.stream(initialState, config);
            // for await (const event of stream) {
            //     console.log("Event:", JSON.stringify(event, null, 2));
            // }

            // Using invoke to get the final state
            finalState = (await this.compiledGraph.invoke(initialState, config)) as EvaluatorOptimizerState; // Explicit cast

            console.log(`--- Invocation Complete for Thread ${threadId} ---`);
            console.log('Final State:', finalState); // Log the entire final state

            if (finalState?.evaluationResult === 'accept' && finalState?.currentSubjectLine) {
                return {
                    finalSubjectLine: finalState.currentSubjectLine,
                    feedback: finalState.feedback,
                    iterations: finalState.iterations,
                };
            } else if (finalState?.iterations >= finalState?.maxIterations) {
                return {
                    error: `Failed to generate an acceptable subject line within ${finalState.maxIterations} iterations.`,
                    lastSubjectLine: finalState?.currentSubjectLine,
                    lastFeedback: finalState?.feedback,
                    iterations: finalState.iterations,
                };
            } else {
                const lastMessage = finalState?.messages?.[finalState.messages.length - 1]?.content ?? 'Unknown error state.';
                return {
                    error: `An unexpected error occurred or evaluation failed. Last known step: ${lastMessage}`,
                    lastSubjectLine: finalState?.currentSubjectLine,
                    lastFeedback: finalState?.feedback,
                    iterations: finalState?.iterations ?? 0,
                };
            }
        } catch (error) {
            console.error(`--- Error Invoking Graph for Thread ${threadId}:`, error);
            return { error: `An error occurred during the process: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
}

export const evaluatorOptimizerAgent = new EvaluatorOptimizerAgent();
