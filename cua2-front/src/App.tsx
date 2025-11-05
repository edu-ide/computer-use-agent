import { useMemo } from 'react';
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider, CssBaseline } from '@mui/material';
import getTheme from './theme';
import Welcome from "./pages/Welcome";
import Task from "./pages/Task";
import { useAgentStore, selectIsDarkMode } from './stores/agentStore';
import { useAgentWebSocket } from './hooks/useAgentWebSocket';
import { config } from './config';

const App = () => {
  const isDarkMode = useAgentStore(selectIsDarkMode);
  const theme = useMemo(() => getTheme(isDarkMode ? 'dark' : 'light'), [isDarkMode]);

  // Initialize WebSocket connection at app level so it persists across route changes
  useAgentWebSocket({ url: config.wsUrl });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Welcome />} />
          <Route path="/task" element={<Task />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default App;
