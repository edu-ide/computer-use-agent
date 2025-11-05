import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useSendTask } from '@/hooks/useSendTask';
import { Box } from '@mui/material';
import { WelcomeScreen } from '@/components';
import { useAgentStore, selectIsConnected } from '@/stores/agentStore';

const Welcome = () => {
  const navigate = useNavigate();
  const isConnected = useAgentStore(selectIsConnected);
  const sendTask = useSendTask();

  const handleSendNewTask = (instruction: string, modelId: string) => {
    sendTask(instruction, modelId);
    // Navigate to task page after starting task
    navigate('/task');
  };

  return (
    <Box
      sx={{
        height: '100vh',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'background.default',
        position: 'relative',
      }}
    >
      <WelcomeScreen onStartTask={handleSendNewTask} isConnected={isConnected} />
    </Box>
  );
};

export default Welcome;
