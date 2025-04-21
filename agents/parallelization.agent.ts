import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages';
import { JsonOutputParser, StringOutputParser } from '@langchain/core/output_parsers';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableConfig } from '@langchain/core/runnables';
import { CompiledStateGraph, END, MemorySaver, START, StateGraph, StateGraphArgs } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';

/**
 * Use Case: Multi-Perspective Customer Review Analysis
 *
 * This agent demonstrates the Parallelization pattern (specifically, Sectioning)
 * by analyzing a customer review simultaneously from multiple perspectives.
 *
 * Workflow:
 * 1. Input: Takes a customer review text.
 * 2. Parallel Analysis: Three separate LLM calls run in parallel:
 *    - Analyze Sentiment: Determines the overall sentiment (e.g., positive, negative, neutral).
 *    - Extract Key Topics: Identifies the main subjects discussed in the review.
 *    - Identify Actionable Suggestions: Extracts any suggestions for improvement mentioned.
 * 3. Aggregation: The results from the parallel analyses are collected.
 * 4. Output: A structured summary containing the sentiment, topics, and suggestions is generated.
 *
 * This approach allows for faster processing compared to sequential analysis and enables
 * focused LLM calls for each specific aspect, potentially improving the quality of each analysis.
 */

/**
 *         / => analyze_sentiment    \
 *   START - => extract_topics       - => aggregate_results => END
 *         \ => identify_suggestions /
 *
 */

// Define the state schema for the graph
interface AgentState {
    review: string;
    sentiment: string | null;
    topics: string[] | null;
    suggestions: string[] | null;
    final_analysis: Record<string, any> | null; // To store the aggregated result
    messages: BaseMessage[]; // To keep track of the conversation history for the checkpointer
}

// Define the structure for the state channels
const agentState: StateGraphArgs<AgentState>['channels'] = {
    review: {
        value: (x?: string, y?: string) => y ?? x ?? '', // Always update review if provided
        default: () => '',
    },
    sentiment: {
        value: (x: string | null, y: string | null) => y ?? x, // Update sentiment
        default: () => null,
    },
    topics: {
        value: (x: string[] | null, y: string[] | null) => y ?? x, // Update topics
        default: () => null,
    },
    suggestions: {
        value: (x: string[] | null, y: string[] | null) => y ?? x, // Update suggestions
        default: () => null,
    },
    final_analysis: {
        value: (x: Record<string, any> | null, y: Record<string, any> | null) => y ?? x, // Update final analysis
        default: () => null,
    },
    messages: {
        value: (x: BaseMessage[], y: BaseMessage[]) => x.concat(y), // Append messages
        default: () => [],
    },
};

// Define output schemas for parallel branches if needed (using parsers directly here)
interface TopicsOutput {
    topics: string[];
}
interface SuggestionsOutput {
    suggestions: string[];
}

class ParallelizationAgent {
    private graph: StateGraph<AgentState>;
    private compiledGraph: CompiledStateGraph<AgentState, Partial<AgentState>>;
    private checkpointer = new MemorySaver();
    private llm = new ChatOpenAI({ model: 'gpt-4o', temperature: 0 });

    constructor() {
        this.graph = new StateGraph({ channels: agentState });

        // Define Nodes for parallel tasks and aggregation
        this.graph.addNode('analyze_sentiment', this.analyzeSentiment);
        this.graph.addNode('extract_topics', this.extractTopics);
        this.graph.addNode('identify_suggestions', this.identifySuggestions);
        this.graph.addNode('aggregate_results', this.aggregateResults);

        // Define Edges
        // Start branches out to the three parallel analysis nodes
        this.graph.addEdge(START, 'analyze_sentiment' as any); // Use 'as any'
        this.graph.addEdge(START, 'extract_topics' as any); // Use 'as any'
        this.graph.addEdge(START, 'identify_suggestions' as any); // Use 'as any'

        // Edges from parallel nodes to the aggregator node
        // The aggregator will wait until all incoming edges are resolved
        this.graph.addEdge('analyze_sentiment' as any, 'aggregate_results' as any); // Use 'as any'
        this.graph.addEdge('extract_topics' as any, 'aggregate_results' as any); // Use 'as any'
        this.graph.addEdge('identify_suggestions' as any, 'aggregate_results' as any); // Use 'as any'

        // Edge from aggregator to END
        this.graph.addEdge('aggregate_results' as any, END); // Use 'as any'

        // Compile the graph
        this.compiledGraph = this.graph.compile({ checkpointer: this.checkpointer });
    }

    // --- Node Functions ---

    private analyzeSentiment = async (state: AgentState): Promise<Partial<AgentState>> => {
        console.log('--- Analyzing Sentiment ---');
        const prompt = PromptTemplate.fromTemplate(
            `Analyze the sentiment of the following customer review. Respond with only one word: "positive", "negative", or "neutral".
            Review:
            {review}`,
        );
        const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
        const sentiment = await chain.invoke({ review: state.review });
        console.log('Sentiment Result:', sentiment);
        const humanMessage = new HumanMessage('Analyze sentiment');
        const aiMessage = new AIMessage({ content: `Sentiment analyzed as: ${sentiment}` });
        return { sentiment: sentiment, messages: [humanMessage, aiMessage] };
    };

    private extractTopics = async (state: AgentState): Promise<Partial<AgentState>> => {
        console.log('--- Extracting Topics ---');
        const prompt = PromptTemplate.fromTemplate(
            `Extract the key topics or subjects discussed in the following customer review.
            Respond with a JSON object containing a single key "topics" which is an array of strings.
            Example format: {{"topics": ["customer service", "product quality", "delivery time"]}}
            Review:
            {review}`,
        );
        const parser = new JsonOutputParser<TopicsOutput>();
        const chain = prompt.pipe(this.llm).pipe(parser);
        let topics: string[] = [];
        let aiMessageContent = '';
        try {
            const result = await chain.invoke({ review: state.review });
            topics = result.topics;
            aiMessageContent = `Topics extracted: ${topics.join(', ')}`;
            console.log('Topics Result:', topics);
        } catch (error) {
            console.error('Topic extraction failed:', error);
            aiMessageContent = `Topic extraction failed: ${error}`;
            topics = ['Error extracting topics'];
        }
        const humanMessage = new HumanMessage('Extract topics');
        const aiMessage = new AIMessage({ content: aiMessageContent });
        return { topics: topics, messages: [humanMessage, aiMessage] };
    };

    private identifySuggestions = async (state: AgentState): Promise<Partial<AgentState>> => {
        console.log('--- Identifying Suggestions ---');
        const prompt = PromptTemplate.fromTemplate(
            `Identify any actionable suggestions for improvement mentioned in the following customer review.
            If no suggestions are found, respond with an empty list.
            Respond with a JSON object containing a single key "suggestions" which is an array of strings.
            Example format: {{"suggestions": ["Improve documentation", "Offer faster shipping"]}}
            Review:
            {review}`,
        );
        const parser = new JsonOutputParser<SuggestionsOutput>();
        const chain = prompt.pipe(this.llm).pipe(parser);
        let suggestions: string[] = [];
        let aiMessageContent = '';
        try {
            const result = await chain.invoke({ review: state.review });
            suggestions = result.suggestions;
            aiMessageContent = suggestions.length > 0 ? `Suggestions identified: ${suggestions.join(', ')}` : 'No suggestions identified.';
            console.log('Suggestions Result:', suggestions);
        } catch (error) {
            console.error('Suggestion identification failed:', error);
            aiMessageContent = `Suggestion identification failed: ${error}`;
            suggestions = ['Error identifying suggestions'];
        }
        const humanMessage = new HumanMessage('Identify suggestions');
        const aiMessage = new AIMessage({ content: aiMessageContent });
        return { suggestions: suggestions, messages: [humanMessage, aiMessage] };
    };

    // Aggregator node using RunnableLambda for simplicity
    private aggregateResults = async (state: AgentState): Promise<Partial<AgentState>> => {
        console.log('--- Aggregating Results ---');
        const final_analysis = {
            sentiment: state.sentiment ?? 'N/A',
            topics: state.topics ?? [],
            suggestions: state.suggestions ?? [],
        };
        console.log('Final Analysis:', final_analysis);
        const humanMessage = new HumanMessage('Aggregate results');
        const aiMessage = new AIMessage({ content: `Analysis complete: ${JSON.stringify(final_analysis)}` });
        return { final_analysis: final_analysis, messages: [humanMessage, aiMessage] };
    };

    // --- Public Invocation Method ---

    async invoke(threadId: string, review: string): Promise<Record<string, any>> {
        const initialState: AgentState = {
            review: review,
            sentiment: null,
            topics: null,
            suggestions: null,
            final_analysis: null,
            messages: [new HumanMessage(review)], // Start with the user's review
        };

        const config: RunnableConfig = { configurable: { thread_id: threadId } };
        let finalState: AgentState | undefined;

        try {
            // Or just invoke and get the final state
            finalState = (await this.compiledGraph.invoke(initialState, config)) as AgentState; // Explicit cast

            if (finalState?.final_analysis) {
                console.log(`--- Invocation Complete for Thread ${threadId} ---`);
                return finalState.final_analysis;
            } else {
                console.error(`--- Invocation Failed for Thread ${threadId}: No final analysis ---`);
                return { error: 'An unexpected error occurred, and no final analysis was generated.' };
            }
        } catch (error) {
            console.error(`--- Error Invoking Graph for Thread ${threadId}:`, error);
            return { error: `An error occurred while processing your request: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
}

export const parallelizationAgent = new ParallelizationAgent();
