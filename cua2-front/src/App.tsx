import { CssBaseline, ThemeProvider } from '@mui/material';
import { useMemo } from 'react';
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { getWebSocketUrl } from './config/api';
import { useAgentWebSocket } from './hooks/useAgentWebSocket';
import Task from "./pages/Task";
import Welcome from "./pages/Welcome";
import Products from "./pages/Products";
import WorkflowList from "./pages/WorkflowList";
import WorkflowDetail from "./pages/WorkflowDetail";
import { Library } from "./pages/Library";
import { selectIsDarkMode, useAgentStore } from './stores/agentStore';
import getTheme from './theme';

const App = () => {
  const isDarkMode = useAgentStore(selectIsDarkMode);
  const theme = useMemo(() => getTheme(isDarkMode ? 'dark' : 'light'), [isDarkMode]);

  // Initialize WebSocket connection at app level so it persists across route changes
  const { stopCurrentTask } = useAgentWebSocket({ url: getWebSocketUrl() });

  // Store functions in window for global access
  (window as Window & { __stopCurrentTask?: () => void }).__stopCurrentTask = stopCurrentTask;


  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<WorkflowList />} />
          <Route path="/welcome" element={<Welcome />} />
          <Route path="/task" element={<Task />} />
          <Route path="/products" element={<Products />} />
          <Route path="/workflows" element={<WorkflowList />} />
          <Route path="/workflows/:workflowId" element={<WorkflowDetail />} />
          <Route path="/library" element={<Library />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default App;
