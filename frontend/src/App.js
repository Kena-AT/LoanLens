import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box, AppBar, Toolbar, Typography, Container, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import SettingsIcon from '@mui/icons-material/Settings';

import Dashboard from './components/Dashboard';
import Predict from './components/Predict';
import Models from './components/Models';
import Monitoring from './components/Monitoring';

function App() {
  const navigate = useNavigate();

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static" elevation={1}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            📊 LoanLens 2.0
          </Typography>
          <Button color="inherit" startIcon={<DashboardIcon />} onClick={() => navigate('/')}>
            Dashboard
          </Button>
          <Button color="inherit" startIcon={<AnalyticsIcon />} onClick={() => navigate('/predict')}>
            Predict
          </Button>
          <Button color="inherit" startIcon={<SettingsIcon />} onClick={() => navigate('/models')}>
            Models
          </Button>
          <Button color="inherit" onClick={() => navigate('/monitoring')}>
            Monitoring
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4, flex: 1 }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/models" element={<Models />} />
          <Route path="/monitoring" element={<Monitoring />} />
        </Routes>
      </Container>

      <Box component="footer" sx={{ py: 3, bgcolor: 'background.paper', mt: 'auto' }}>
        <Typography variant="body2" color="text.secondary" align="center">
          LoanLens 2.0 - Advanced Loan Default Prediction System © 2024
        </Typography>
      </Box>
    </Box>
  );
}

export default App;
