import { Header, SandboxViewer, StepsList, Timeline } from '@/components';
import { selectIsAgentProcessing, selectMetadata, selectSelectedStep, selectTrace, selectVncUrl, useAgentStore } from '@/stores/agentStore';
import { Box } from '@mui/material';
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const Task = () => {
  const navigate = useNavigate();

  // Get state from Zustand store
  const trace = useAgentStore(selectTrace);
  const isAgentProcessing = useAgentStore(selectIsAgentProcessing);
  const vncUrl = useAgentStore(selectVncUrl);
  const metadata = useAgentStore(selectMetadata);
  const selectedStep = useAgentStore(selectSelectedStep);
  const error = useAgentStore((state) => state.error);

  // Redirect to home if no trace is present
  useEffect(() => {
    if (!trace) {
      console.log('No trace found, redirecting to home...');
      navigate('/', { replace: true });
    }
  }, [trace, navigate]);

  // Handler for going back to home
  const handleBackToHome = () => {
    const currentTrace = useAgentStore.getState().trace;

    // Stop the current task if it's running
    const stopTask = (window as Window & { __stopCurrentTask?: () => void }).__stopCurrentTask;
    if (stopTask) {
      stopTask();
    }

    // Reset frontend state
    useAgentStore.getState().resetAgent();

    // Reload the page to reconnect websocket
    window.location.href = '/';
  };

  // Determine if we should show success/fail status (same logic as SandboxViewer)
  const showStatus = !trace?.isRunning && !selectedStep && metadata && metadata.numberOfSteps > 0;

  // Don't render anything if no trace (will redirect)
  if (!trace) {
    return null;
  }

  return (
    <Box
      sx={{
        height: '100vh',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'background.default',
      }}
    >
      {/* Header */}
      <Header
        isAgentProcessing={isAgentProcessing}
        onBackToHome={handleBackToHome}
      />

      {/* Main Content */}
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'stretch',
          minHeight: 0,
          p: 0,
          overflowY: 'auto',
          overflowX: 'hidden',
        }}
      >
        <Box
          sx={{
            width: '100%',
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            p: { xs: 2, md: 4 },
            pb: { xs: 2, md: 3 },
          }}
        >
          {/* Left Side: OS Stream + Metadata */}
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              minWidth: 0,
              pr: { xs: 0, md: 1.5 },
              gap: { xs: 2, md: 3 },
              overflow: 'visible',
            }}
          >
            {/* Sandbox Viewer */}
            <SandboxViewer
              vncUrl={vncUrl}
              isAgentProcessing={isAgentProcessing}
              metadata={metadata}
              traceStartTime={trace?.timestamp}
              selectedStep={selectedStep}
              isRunning={trace?.isRunning || false}
            />

            {/* Timeline - Always show, even with default values */}
            <Timeline
              metadata={metadata && metadata.maxSteps > 0 ? metadata : {
                traceId: trace?.id || '',
                inputTokensUsed: metadata?.inputTokensUsed || 0,
                outputTokensUsed: metadata?.outputTokensUsed || 0,
                duration: metadata?.duration || 0,
                numberOfSteps: metadata?.numberOfSteps || 0,
                maxSteps: 200, // Default max steps (will be updated by backend)
                completed: metadata?.completed || false,
              }}
              isRunning={trace?.isRunning || false}
            />
          </Box>

          {/* Right Side: Steps List */}
          <StepsList trace={trace} />
        </Box>
      </Box>
    </Box>
  );
};

export default Task;
