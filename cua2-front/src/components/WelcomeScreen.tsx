import { fetchAvailableModels, generateRandomQuestion } from '@/services/api';
import { selectAvailableModels, selectIsDarkMode, selectIsLoadingModels, selectSelectedModelId, useAgentStore } from '@/stores/agentStore';
import DarkModeOutlined from '@mui/icons-material/DarkModeOutlined';
import LightModeOutlined from '@mui/icons-material/LightModeOutlined';
import SendIcon from '@mui/icons-material/Send';
import ShuffleIcon from '@mui/icons-material/Shuffle';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import { Box, Button, CircularProgress, Container, FormControl, IconButton, InputLabel, MenuItem, Paper, Select, TextField, Typography } from '@mui/material';
import React, { useEffect, useRef, useState } from 'react';

interface WelcomeScreenProps {
  onStartTask: (instruction: string, modelId: string) => void;
  isConnected: boolean;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onStartTask, isConnected }) => {
  const [customTask, setCustomTask] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isGeneratingQuestion, setIsGeneratingQuestion] = useState(false);
  const typingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const isDarkMode = useAgentStore(selectIsDarkMode);
  const toggleDarkMode = useAgentStore((state) => state.toggleDarkMode);
  const selectedModelId = useAgentStore(selectSelectedModelId);
  const setSelectedModelId = useAgentStore((state) => state.setSelectedModelId);
  const availableModels = useAgentStore(selectAvailableModels);
  const isLoadingModels = useAgentStore(selectIsLoadingModels);
  const setAvailableModels = useAgentStore((state) => state.setAvailableModels);
  const setIsLoadingModels = useAgentStore((state) => state.setIsLoadingModels);

  // Load available models on mount
  useEffect(() => {
    const loadModels = async () => {
      setIsLoadingModels(true);
      try {
        const models = await fetchAvailableModels();
        setAvailableModels(models);

        // Set first model as default if current selection is not in the list
        if (models.length > 0 && !models.includes(selectedModelId)) {
          setSelectedModelId(models[0]);
        }
      } catch (error) {
        console.error('Failed to load models:', error);
        // Fallback to empty array on error
        setAvailableModels([]);
      } finally {
        setIsLoadingModels(false);
      }
    };

    loadModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Clean up typing interval on unmount
  useEffect(() => {
    return () => {
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
      }
    };
  }, []);

  const handleWriteRandomTask = async () => {
    // Clear any existing typing interval
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
      typingIntervalRef.current = null;
    }

    setIsGeneratingQuestion(true);
    try {
      const randomTask = await generateRandomQuestion();

      // Clear current text
      setCustomTask('');
      setIsTyping(true);

      // Type effect
      let currentIndex = 0;
      typingIntervalRef.current = setInterval(() => {
        if (currentIndex < randomTask.length) {
          setCustomTask(randomTask.substring(0, currentIndex + 1));
          currentIndex++;
        } else {
          if (typingIntervalRef.current) {
            clearInterval(typingIntervalRef.current);
            typingIntervalRef.current = null;
          }
          setIsTyping(false);
        }
      }, 10); // 10ms per character
    } catch (error) {
      console.error('Failed to generate question:', error);
      setIsTyping(false);
    } finally {
      setIsGeneratingQuestion(false);
    }
  };

  const handleCustomTask = () => {
    if (customTask.trim() && !isTyping) {
      onStartTask(customTask.trim(), selectedModelId);
    }
  };

  return (
    <>
      {/* Dark Mode Toggle - Top Right (Absolute to viewport) */}
      <Box sx={{ position: 'absolute', top: 24, right: 24, zIndex: 1000 }}>
        <IconButton
          onClick={toggleDarkMode}
          size="medium"
          sx={{
            color: 'text.primary',
            backgroundColor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider',
            '&:hover': {
              backgroundColor: 'action.hover',
              borderColor: 'primary.main',
            },
          }}
        >
          {isDarkMode ? <LightModeOutlined /> : <DarkModeOutlined />}
        </IconButton>
      </Box>

      <Container
        maxWidth="md"
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          textAlign: 'center',
          py: 8,
        }}
      >
        {/* Hugging Face Logo */}
        <Box
          component="img"
          src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
          alt="Hugging Face"
          sx={{
            width: 96,
            height: 96,
            mb: 3,
            flexShrink: 0,
            display: 'block',
            maxWidth: '100%',
          }}
        />

        {/* Title */}
        <Typography
          variant="h2"
          sx={{
            fontWeight: 800,
            mb: 1,
            color: 'text.primary',
          }}
        >
          Computer Use Agent
        </Typography>

        {/* Powered by smolagents and E2B */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            mb: 2,
            flexWrap: 'wrap',
            justifyContent: 'center',
          }}
        >
          <Typography
            variant="body2"
            sx={{
              color: 'text.secondary',
              fontWeight: 500,
            }}
          >
            Powered by
          </Typography>

          {/* smolagents link */}
          <Box
            component="a"
            href="https://github.com/huggingface/smolagents"
            target="_blank"
            rel="noopener noreferrer"
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.75,
              textDecoration: 'none',
              transition: 'all 0.2s ease',
              '&:hover': {
                '& .smolagents-text': {
                  textDecoration: 'underline',
                },
              },
            }}
          >
            {/* Hugging Face Official Logo */}
            <Box
              component="img"
              src="https://cdn-avatars.huggingface.co/v1/production/uploads/63d10d4e8eaa4831005e92b5/a3R8vs2eGE578q4LEpaHB.png"
              alt="Hugging Face"
              sx={{
                width: 24,
                height: 24,
                flexShrink: 0,
              }}
            />

            <Typography
              className="smolagents-text"
              sx={{
                color: 'primary.main',
                fontWeight: 700,
                fontSize: '1rem',
              }}
            >
              smolagents
            </Typography>

            {/* GitHub stars badge */}
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                px: 1,
                py: 0.25,
                backgroundColor: (theme) =>
                  theme.palette.mode === 'dark'
                    ? 'rgba(144, 202, 249, 0.08)'
                    : 'rgba(25, 118, 210, 0.08)',
                borderRadius: 1,
                border: '1px solid',
                borderColor: 'primary.main',
              }}
            >
              <Box component="span" sx={{ fontSize: '0.75rem' }}>⭐</Box>
              <Typography
                variant="caption"
                sx={{
                  fontWeight: 700,
                  color: 'primary.main',
                  fontSize: '0.75rem',
                }}
              >
                23.7k
              </Typography>
            </Box>
          </Box>

          {/* Separator */}
          <Typography
            variant="body2"
            sx={{
              color: 'text.secondary',
              mx: 0.5,
            }}
          >
            &
          </Typography>

          {/* E2B link */}
          <Box
            component="a"
            href="https://e2b.dev/"
            target="_blank"
            rel="noopener noreferrer"
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.75,
              textDecoration: 'none',
              transition: 'all 0.2s ease',
              '&:hover': {
                '& .e2b-text': {
                  textDecoration: 'underline',
                },
              },
            }}
          >
            {/* E2B Logo */}
            <Box
              component="img"
              src="https://avatars.githubusercontent.com/u/129434473?s=200&v=4"
              alt="E2B"
              sx={{
                width: 24,
                height: 24,
                flexShrink: 0,
                borderRadius: '50%',
              }}
            />

            <Typography
              className="e2b-text"
              sx={{
                color: 'primary.main',
                fontWeight: 700,
                fontSize: '1rem',
              }}
            >
              E2B
            </Typography>
          </Box>
        </Box>

        {/* Subtitle */}
        <Typography
          variant="h6"
          sx={{
            color: 'text.secondary',
            fontWeight: 500,
            mb: 1,
          }}
        >
          AI-Powered Computer Use Automation
        </Typography>

        {/* Description */}
        <Typography
          variant="body1"
          sx={{
            color: 'text.secondary',
            maxWidth: '650px',
            mb: 3,
            lineHeight: 1.7,
          }}
        >
          Experience the future of AI automation as agents operate computers in real time to complete complex on-screen tasks (GUI agents).
          Built by{' '}
          <Box
            component="a"
            href="https://huggingface.co"
            target="_blank"
            rel="noopener noreferrer"
            sx={{
              color: 'primary.main',
              textDecoration: 'none',
              fontWeight: 700,
              '&:hover': {
                textDecoration: 'underline',
              },
            }}
          >
            Hugging Face
          </Box>
          , this platform provides intuitive <strong>visualization and annotation tools</strong>, enabling <strong>manual preferential data annotation</strong> for advanced agentic AI research.
        </Typography>

        {/* Task Input Section */}
        <Paper
          elevation={0}
          sx={{
            maxWidth: '725px',
            width: '100%',
            p: 2.5,
            border: '2px solid',
            borderColor: isConnected ? 'primary.main' : 'divider',
            borderRadius: 2,
            backgroundColor: 'background.paper',
            transition: 'all 0.2s ease',
            '&:hover': isConnected ? {
              borderColor: 'primary.dark',
              boxShadow: (theme) => `0 4px 16px ${theme.palette.mode === 'dark' ? 'rgba(79, 134, 198, 0.3)' : 'rgba(79, 134, 198, 0.15)'}`,
            } : {},
          }}
        >
          {/* Input Field */}
          <TextField
            fullWidth
            placeholder="Describe your task here..."
            value={customTask}
            onChange={(e) => setCustomTask(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && isConnected && customTask.trim() && !isTyping) {
                handleCustomTask();
              }
            }}
            disabled={!isConnected || isTyping}
            multiline
            rows={3}
            sx={{
              mb: 2,
              '& .MuiOutlinedInput-root': {
                borderRadius: 1.5,
                backgroundColor: 'action.hover',
                color: 'text.primary',
                '& fieldset': {
                  borderColor: 'divider',
                },
                '&:hover fieldset': {
                  borderColor: 'text.secondary',
                },
                '&.Mui-focused fieldset': {
                  borderColor: 'primary.main',
                  borderWidth: '2px',
                },
              },
              '& .MuiInputBase-input': {
                color: (theme) => theme.palette.mode === 'dark' ? '#FFFFFF !important' : '#000000 !important',
                fontWeight: 500,
                WebkitTextFillColor: (theme) => theme.palette.mode === 'dark' ? '#FFFFFF !important' : '#000000 !important',
              },
              '& .MuiInputBase-input.Mui-disabled': {
                color: (theme) => theme.palette.mode === 'dark' ? '#FFFFFF !important' : '#000000 !important',
                WebkitTextFillColor: (theme) => theme.palette.mode === 'dark' ? '#FFFFFF !important' : '#000000 !important',
              },
              '& .MuiInputBase-input::placeholder': {
                color: 'text.secondary',
                opacity: 0.7,
              },
            }}
          />

          {/* Model Selection + Buttons Row */}
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', justifyContent: 'space-between' }}>
            {/* Model Select */}
            <FormControl size="small" sx={{ minWidth: 240 }}>
              <InputLabel id="model-select-label">Model</InputLabel>
              <Select
                labelId="model-select-label"
                value={availableModels.length > 0 && availableModels.includes(selectedModelId) ? selectedModelId : ''}
                label="Model"
                onChange={(e) => setSelectedModelId(e.target.value)}
                disabled={!isConnected || isTyping || isLoadingModels}
                sx={{
                  borderRadius: 1.5,
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderWidth: 2,
                  },
                }}
              >
                {isLoadingModels ? (
                  <MenuItem disabled>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CircularProgress size={16} />
                      <Typography variant="body2">Loading models...</Typography>
                    </Box>
                  </MenuItem>
                ) : availableModels.length === 0 ? (
                  <MenuItem disabled>
                    <Typography variant="body2" sx={{ color: 'error.main' }}>
                      No models available
                    </Typography>
                  </MenuItem>
                ) : (
                  availableModels.map((modelId) => (
                    <MenuItem key={modelId} value={modelId}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <SmartToyIcon sx={{ fontSize: '0.9rem', color: 'primary.main' }} />
                        <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.875rem' }}>
                          {modelId.split('/').pop()}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))
                )}
              </Select>
            </FormControl>

            {/* Buttons on the right */}
            <Box sx={{ display: 'flex', gap: 1.5 }}>
              <Button
                variant="outlined"
                onClick={handleWriteRandomTask}
                disabled={!isConnected || isTyping || isGeneratingQuestion}
                startIcon={isGeneratingQuestion ? <CircularProgress size={16} /> : <ShuffleIcon />}
                sx={{
                  borderRadius: 1.5,
                  textTransform: 'none',
                  fontWeight: 600,
                  borderWidth: 2,
                  px: 3,
                  '&:hover': {
                    borderWidth: 2,
                  },
                }}
              >
                {isGeneratingQuestion ? 'Generating...' : isTyping ? 'Writing...' : 'Write random task'}
              </Button>

              <Button
                variant="contained"
                onClick={handleCustomTask}
                disabled={!isConnected || !customTask.trim() || isTyping}
                sx={{
                  borderRadius: 1.5,
                  textTransform: 'none',
                  fontWeight: 600,
                  px: 4,
                  background: 'linear-gradient(135deg, #4F86C6 0%, #2B5C94 100%)',
                }}
                endIcon={<SendIcon />}
              >
                Run Task
              </Button>
            </Box>
          </Box>
        </Paper>

        {/* Research Notice */}
        <Typography
          variant="body2"
          sx={{
            color: 'text.secondary',
            maxWidth: '700px',
            mt: 3,
            mb: 2,
            lineHeight: 1.6,
            fontStyle: 'italic',
            opacity: 0.8,
            textAlign: 'center',
          }}
        >
          Please be aware that by using the demo, you agree that the traces are stored for research purposes.
          <strong>Please do not write any personal information.</strong>
        </Typography>

        {/* Connection status hint */}
        {!isConnected && (
          <Typography
            variant="caption"
            sx={{
              mt: 2,
              color: 'text.secondary',
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: 'warning.main',
                animation: 'pulse 2s ease-in-out infinite',
                '@keyframes pulse': {
                  '0%, 100%': { opacity: 1 },
                  '50%': { opacity: 0.5 },
                },
              }}
            />
            Make sure the backend is running on port 8000
          </Typography>
        )}
      </Container>
    </>
  );
};
