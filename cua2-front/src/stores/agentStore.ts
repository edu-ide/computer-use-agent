import { AgentStep, AgentTrace, AgentTraceMetadata, FinalStep } from '@/types/agent';
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

interface AgentState {
  // State
  trace?: AgentTrace;
  traceId: string | null; // Set by backend heartbeat, persists during connection
  isAgentProcessing: boolean;
  isConnectingToE2B: boolean; // New state for E2B connection
  vncUrl: string;
  selectedModelId: string;
  availableModels: string[];
  isLoadingModels: boolean;
  isConnected: boolean;
  error?: string;
  isDarkMode: boolean;
  selectedStepIndex: number | null; // null = live mode, number = viewing specific step or 'final'
  finalStep?: FinalStep; // Special step for success/failure

  // Actions
  setTrace: (trace: AgentTrace | undefined) => void;
  setTraceId: (traceId: string | null) => void;
  updateTraceWithStep: (step: AgentStep, metadata: AgentTraceMetadata) => void;
  updateStepEvaluation: (stepId: string, evaluation: 'like' | 'dislike' | 'neutral') => void;
  updateTraceEvaluation: (evaluation: 'success' | 'failed' | 'not_evaluated') => void;
  completeTrace: (metadata: AgentTraceMetadata, finalState?: 'success' | 'stopped' | 'max_steps_reached' | 'error' | 'sandbox_timeout') => void;
  setIsAgentProcessing: (processing: boolean) => void;
  setIsConnectingToE2B: (connecting: boolean) => void;
  setVncUrl: (url: string) => void;
  setSelectedModelId: (modelId: string) => void;
  setAvailableModels: (models: string[]) => void;
  setIsLoadingModels: (loading: boolean) => void;
  setIsConnected: (connected: boolean) => void;
  setError: (error: string | undefined) => void;
  setSelectedStepIndex: (index: number | null) => void;
  toggleDarkMode: () => void;
  resetAgent: () => void;
}

const initialState = {
  trace: undefined,
  traceId: null, // Will be set by backend heartbeat
  isAgentProcessing: false,
  isConnectingToE2B: false,
  vncUrl: '',
  selectedModelId: 'Qwen/Qwen3-VL-30B-A3B-Instruct',
  availableModels: [],
  isLoadingModels: false,
  isConnected: false,
  error: undefined,
  isDarkMode: false,
  selectedStepIndex: null, // null = live mode
  finalStep: undefined,
};

export const useAgentStore = create<AgentState>()(
  devtools(
    (set) => ({
      ...initialState,

      // Set the complete trace
      setTrace: (trace) =>
        set({ trace }, false, 'setTrace'),

      // Set trace ID (set by backend heartbeat, only cleared on disconnect)
      setTraceId: (traceId) =>
        set({ traceId }, false, 'setTraceId'),

      // Update trace with a new step
      updateTraceWithStep: (step, metadata) =>
        set(
          (state) => {
            if (!state.trace) return state;

            const existingSteps = state.trace.steps || [];
            const stepExists = existingSteps.some((s) => s.stepId === step.stepId);

            if (stepExists) return state;

            // Preserve existing maxSteps if new metadata has 0
            const updatedMetadata = {
              ...metadata,
              maxSteps: metadata.maxSteps > 0
                ? metadata.maxSteps
                : (state.trace.traceMetadata?.maxSteps || 200),
            };

            return {
              trace: {
                ...state.trace,
                steps: [...existingSteps, step],
                traceMetadata: updatedMetadata,
                isRunning: true,
              },
            };
          },
          false,
          'updateTraceWithStep'
        ),

      // Update step evaluation in the store
      updateStepEvaluation: (stepId, evaluation) =>
        set(
          (state) => {
            if (!state.trace || !state.trace.steps) return state;

            const updatedSteps = state.trace.steps.map((step) =>
              step.stepId === stepId
                ? { ...step, step_evaluation: evaluation }
                : step
            );

            return {
              trace: {
                ...state.trace,
                steps: updatedSteps,
              },
            };
          },
          false,
          'updateStepEvaluation'
        ),

      // Update trace evaluation in the store
      updateTraceEvaluation: (evaluation) =>
        set(
          (state) => {
            if (!state.trace || !state.trace.traceMetadata) return state;

            const updatedMetadata = {
              ...state.trace.traceMetadata,
              user_evaluation: evaluation,
            };

            return {
              trace: {
                ...state.trace,
                traceMetadata: updatedMetadata,
              },
              // Also update finalStep metadata if it exists
              finalStep: state.finalStep ? {
                ...state.finalStep,
                metadata: {
                  ...state.finalStep.metadata,
                  user_evaluation: evaluation,
                },
              } : state.finalStep,
            };
          },
          false,
          'updateTraceEvaluation'
        ),

      // Complete the trace
      completeTrace: (metadata, finalState?: 'success' | 'stopped' | 'max_steps_reached' | 'error' | 'sandbox_timeout') =>
        set(
          (state) => {
            if (!state.trace) return state;

            // Preserve existing maxSteps if new metadata has 0
            const updatedMetadata = {
              ...metadata,
              maxSteps: metadata.maxSteps > 0
                ? metadata.maxSteps
                : (state.trace.traceMetadata?.maxSteps || 200),
              completed: true,
            };

            // Determine the final step type based on final_state from backend
            let stepType: 'success' | 'failure' | 'stopped' | 'max_steps_reached' | 'sandbox_timeout';
            let stepMessage: string | undefined;

            if (finalState === 'stopped') {
              stepType = 'stopped';
              stepMessage = 'Task stopped by user';
            } else if (finalState === 'max_steps_reached') {
              stepType = 'max_steps_reached';
              stepMessage = 'Maximum steps reached';
            } else if (finalState === 'sandbox_timeout') {
              stepType = 'sandbox_timeout';
              stepMessage = 'Sandbox timeout';
            } else if (finalState === 'error' || state.error) {
              stepType = 'failure';
              stepMessage = state.error || 'Task failed';
            } else {
              stepType = 'success';
              stepMessage = undefined;
            }

            const finalStep: FinalStep = {
              type: stepType,
              message: stepMessage,
              metadata: updatedMetadata,
            };

            return {
              trace: {
                ...state.trace,
                isRunning: false,
                traceMetadata: updatedMetadata,
              },
              finalStep,
              // Keep error in state for display
              selectedStepIndex: null, // Reset to live mode on completion
            };
          },
          false,
          'completeTrace'
        ),

      // Set processing state
      setIsAgentProcessing: (isAgentProcessing) =>
        set({ isAgentProcessing }, false, 'setIsAgentProcessing'),

      // Set E2B connection state
      setIsConnectingToE2B: (isConnectingToE2B) =>
        set({ isConnectingToE2B }, false, 'setIsConnectingToE2B'),

      // Set VNC URL
      setVncUrl: (vncUrl) =>
        set({ vncUrl }, false, 'setVncUrl'),

      // Set selected model ID
      setSelectedModelId: (selectedModelId) =>
        set({ selectedModelId }, false, 'setSelectedModelId'),

      // Set available models
      setAvailableModels: (availableModels) =>
        set({ availableModels }, false, 'setAvailableModels'),

      // Set loading models state
      setIsLoadingModels: (isLoadingModels) =>
        set({ isLoadingModels }, false, 'setIsLoadingModels'),

      // Set connection status
      setIsConnected: (isConnected) =>
        set({ isConnected }, false, 'setIsConnected'),

      // Set error
      setError: (error) =>
        set(
          (state) => {
            // If there's an error and a trace, mark it as failed
            if (error && state.trace) {
              const metadata = state.trace.traceMetadata || {
                traceId: state.trace.id,
                inputTokensUsed: 0,
                outputTokensUsed: 0,
                duration: 0,
                numberOfSteps: state.trace.steps?.length || 0,
                maxSteps: 200,
                completed: false,
                final_state: null,
                user_evaluation: 'not_evaluated' as const,
              };

              // Ensure maxSteps is not 0
              const finalMetadata: AgentTraceMetadata = {
                ...metadata,
                maxSteps: metadata.maxSteps > 0 ? metadata.maxSteps : 200,
                final_state: metadata.final_state || null,
                user_evaluation: metadata.user_evaluation || 'not_evaluated',
              };

              const finalStep: FinalStep = {
                type: 'failure',
                message: error,
                metadata: finalMetadata,
              };

              return {
                error,
                finalStep,
                trace: {
                  ...state.trace,
                  isRunning: false,
                },
                selectedStepIndex: null, // Reset to live mode on error
              };
            }
            return { error };
          },
          false,
          'setError'
        ),

      // Set selected step index for time travel
      setSelectedStepIndex: (selectedStepIndex) =>
        set({ selectedStepIndex }, false, 'setSelectedStepIndex'),

      // Toggle dark mode
      toggleDarkMode: () =>
        set((state) => ({ isDarkMode: !state.isDarkMode }), false, 'toggleDarkMode'),

      // Reset agent state (but preserve traceId from backend during connection)
      resetAgent: () =>
        set((state) => ({
          ...initialState,
          traceId: state.traceId,  // IMPORTANT: Keep traceId from backend
          isDarkMode: state.isDarkMode,  // Keep dark mode preference
          isConnected: state.isConnected,  // Keep connection status
          selectedModelId: state.selectedModelId,  // Keep selected model
          availableModels: state.availableModels,  // Keep available models
          isLoadingModels: state.isLoadingModels  // Keep loading state
        }), false, 'resetAgent'),
    }),
    { name: 'AgentStore' }
  )
);

// Selectors for better performance
export const selectTrace = (state: AgentState) => state.trace;
export const selectTraceId = (state: AgentState) => state.traceId;
export const selectIsAgentProcessing = (state: AgentState) => state.isAgentProcessing;
export const selectIsConnectingToE2B = (state: AgentState) => state.isConnectingToE2B;
export const selectVncUrl = (state: AgentState) => state.vncUrl;
export const selectSelectedModelId = (state: AgentState) => state.selectedModelId;
export const selectAvailableModels = (state: AgentState) => state.availableModels;
export const selectIsLoadingModels = (state: AgentState) => state.isLoadingModels;
export const selectIsConnected = (state: AgentState) => state.isConnected;
export const selectSteps = (state: AgentState) => state.trace?.steps;
export const selectMetadata = (state: AgentState) => state.trace?.traceMetadata;
export const selectError = (state: AgentState) => state.error;
export const selectIsDarkMode = (state: AgentState) => state.isDarkMode;
export const selectSelectedStepIndex = (state: AgentState) => state.selectedStepIndex;
export const selectFinalStep = (state: AgentState) => state.finalStep;

// Composite selector for selected step (avoids infinite loops)
export const selectSelectedStep = (state: AgentState) => {
  const steps = state.trace?.steps;
  const selectedIndex = state.selectedStepIndex;

  if (selectedIndex === null || !steps || selectedIndex >= steps.length) {
    return null;
  }

  return steps[selectedIndex];
};
