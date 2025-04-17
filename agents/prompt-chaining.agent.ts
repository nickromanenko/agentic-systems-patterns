import { BaseMessage, HumanMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableConfig } from '@langchain/core/runnables';
import { CompiledStateGraph, END, MemorySaver, START, StateGraph, StateGraphArgs } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import colors from 'colors';

// Define the state schema for the graph
interface AgentState {
    topic: string;
    outline: string | null;
    story: string | null;
    title: string | null;
    validation_error: string | null;
    messages: BaseMessage[]; // To keep track of the conversation history for the checkpointer
}

const agentState: StateGraphArgs<AgentState>['channels'] = {
    topic: {
        value: (x?: string, y?: string) => y ?? x ?? '', // Always update topic if provided
        default: () => '',
    },
    outline: {
        value: (x: string | null, y: string | null) => y ?? x, // Update outline if provided
        default: () => null,
    },
    story: {
        value: (x: string | null, y: string | null) => y ?? x, // Update story if provided
        default: () => null,
    },
    title: {
        value: (x: string | null, y: string | null) => y ?? x, // Update title if provided
        default: () => null,
    },
    validation_error: {
        value: (x: string | null, y: string | null) => y ?? x, // Update error if provided
        default: () => null,
    },
    messages: {
        value: (x: BaseMessage[], y: BaseMessage[]) => x.concat(y), // Append messages
        default: () => [],
    },
};

class PromptChainingAgent {
    // Simplify graph types, rely on inference
    private graph: StateGraph<AgentState>;
    private compiledGraph: CompiledStateGraph<AgentState, Partial<AgentState>>;
    private checkpointer = new MemorySaver();
    private llm = new ChatOpenAI({ temperature: 0.7, model: 'gpt-4.1' }); // Use a bit of creativity

    constructor() {
        this.graph = new StateGraph({ channels: agentState });

        // Define Nodes
        this.graph.addNode('generate_outline', this.generateOutline.bind(this));
        this.graph.addNode('validate_outline', this.validateOutline.bind(this));
        this.graph.addNode('write_story', this.writeStory.bind(this));
        this.graph.addNode('generate_title', this.generateTitle.bind(this));
        this.graph.addNode('handle_error', this.handleError.bind(this)); // Node to handle validation failure

        // Define Edges
        this.graph.addEdge(START, 'generate_outline' as any);
        this.graph.addEdge('generate_outline' as any, 'validate_outline' as any);

        // Conditional Edge: Based on outline validation
        this.graph.addConditionalEdges('validate_outline' as any, this.decideNextStep.bind(this), {
            continue: 'write_story' as any,
            error: 'handle_error' as any,
        });

        this.graph.addEdge('write_story' as any, 'generate_title' as any);
        this.graph.addEdge('generate_title' as any, END);
        this.graph.addEdge('handle_error' as any, END); // End if validation fails

        // Compile the graph
        this.compiledGraph = this.graph.compile({ checkpointer: this.checkpointer });
    }

    // --- Node Functions ---

    private async generateOutline(state: AgentState): Promise<Partial<AgentState>> {
        console.log(colors.magenta('--- Generating Outline ---'));
        const prompt = ChatPromptTemplate.fromMessages([
            [
                'system',
                'You are a creative assistant. Generate a 3-section outline for a short story based on the following topic. Name the sections like "Section 1: ...".',
            ],
            ['human', 'Topic: {topic}'],
        ]);
        const chain = prompt.pipe(this.llm);
        const outline = await chain.invoke({ topic: state.topic });
        console.log('Generated Outline:', outline.content);
        return { outline: outline.content as string, messages: [new HumanMessage(`Generate outline for: ${state.topic}`), outline] };
    }

    private async validateOutline(state: AgentState): Promise<Partial<AgentState>> {
        console.log(colors.magenta('--- Validating Outline ---'));
        if (!state.outline) {
            return {
                validation_error: 'Outline is missing.',
                messages: [new HumanMessage('Validate outline'), new HumanMessage('Error: Outline missing')],
            };
        }
        // Simple validation: Check if it roughly contains 3 sections (e.g., count lines or keywords)
        const sections = state.outline.split('\n').filter(line => line.trim().length > 0 && /^\d+\.|section|part/i.test(line));
        if (sections.length < 3) {
            console.log('Validation Failed: Not enough sections.');
            return {
                validation_error: `Outline validation failed: Expected at least 3 sections, found ${sections.length}.`,
                messages: [new HumanMessage('Validate outline'), new HumanMessage('Error: Validation Failed')],
            };
        }
        console.log('Validation Passed.');
        return { validation_error: null, messages: [new HumanMessage('Validate outline'), new HumanMessage('Validation Passed')] }; // Clear any previous error
    }

    private decideNextStep(state: AgentState): 'continue' | 'error' {
        console.log(colors.magenta('--- Deciding Next Step ---'));
        if (state.validation_error) {
            console.log('Decision: Error');
            return 'error';
        }
        console.log('Decision: Continue');
        return 'continue';
    }

    private async writeStory(state: AgentState): Promise<Partial<AgentState>> {
        console.log(colors.magenta('--- Writing Story ---'));
        const prompt = ChatPromptTemplate.fromMessages([
            ['system', 'You are a talented story writer. Write a short story based on the following outline.'],
            ['human', 'Outline:\n{outline}'],
        ]);
        const chain = prompt.pipe(this.llm);
        const story = await chain.invoke({ outline: state.outline });
        console.log('Generated Story:', story.content);
        return { story: story.content as string, messages: [new HumanMessage('Write story based on outline'), story] };
    }

    private async generateTitle(state: AgentState): Promise<Partial<AgentState>> {
        console.log(colors.magenta('--- Generating Title ---'));
        const prompt = ChatPromptTemplate.fromMessages([
            ['system', 'Generate a short, catchy title for the following story.'],
            ['human', 'Story:\n{story}'],
        ]);
        const chain = prompt.pipe(this.llm);
        const title = await chain.invoke({ story: state.story });
        console.log('Generated Title:', title.content);
        return { title: title.content as string, messages: [new HumanMessage('Generate title for story'), title] };
    }

    private async handleError(state: AgentState): Promise<Partial<AgentState>> {
        console.log(colors.red('--- Handling Error ---'));
        console.error('Outline validation failed:', colors.red(state.validation_error));
        // The error is already in the state, just log it or prepare a final error message
        return { messages: [new HumanMessage(`Error occurred: ${state.validation_error}`)] };
    }

    // --- Public Invocation Method ---

    async invoke(threadId: string, topic: string): Promise<string> {
        // Need to provide all fields for the initial state, even if null
        const initialState: AgentState = {
            topic: topic,
            outline: null,
            story: null,
            title: null,
            validation_error: null,
            messages: [new HumanMessage(topic)],
        };

        const config: RunnableConfig = { configurable: { thread_id: threadId } };
        const finalState = await this.compiledGraph.invoke(initialState, config);

        if (finalState.validation_error) {
            return `Failed to generate story: ${finalState.validation_error}`;
        }

        if (finalState.title && finalState.story) {
            return `Title: ${finalState.title}\n\nStory:\n${finalState.story}`;
        } else {
            // Should not happen if validation passes, but handle defensively
            return 'An unexpected error occurred during story generation.';
        }
    }
}

export const promptChainingAgent = new PromptChainingAgent();
