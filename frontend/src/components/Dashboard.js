import React, { useEffect, useState } from 'react';
import { Grid, Paper, Typography, Box, Card, CardContent, LinearProgress } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function Dashboard() {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [healthRes, metricsRes, modelsRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/health`),
        axios.get(`${API_BASE_URL}/metrics`),
        axios.get(`${API_BASE_URL}/models`)
      ]);

      setHealth(healthRes.data);
      setMetrics(metricsRes.data);
      setModels(modelsRes.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return <LinearProgress />;
  }

  const systemData = health?.system ? [
    { name: 'CPU', value: health.system.cpu_percent || 0 },
    { name: 'Memory', value: health.system.memory_percent || 0 },
    { name: 'Disk', value: health.system.disk_usage || 0 }
  ] : [];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* System Status */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                System Status
              </Typography>
              <Typography variant="h5" component="div" color={health?.status === 'healthy' ? 'success' : 'error'}>
                {health?.status?.toUpperCase()}
              </Typography>
              <Typography color="textSecondary">
                Uptime: {health?.uptime}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Models Loaded */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Models Loaded
              </Typography>
              <Typography variant="h3" component="div">
                {models?.length || 0}
              </Typography>
              <Typography color="textSecondary">
                Active models ready for prediction
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Predictions Made */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Predictions
              </Typography>
              <Typography variant="h3" component="div">
                {metrics?.prediction_count?.toLocaleString() || 0}
              </Typography>
              <Typography color="textSecondary">
                Error Rate: {((metrics?.error_rate || 0) * 100).toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* System Resources Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              System Resources
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={systemData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#1976d2" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Model Performance */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Model Performance
            </Typography>
            {metrics && (
              <Box>
                <Typography>Average Latency: {metrics.average_latency_ms?.toFixed(2)} ms</Typography>
                <Typography>Recent Predictions: {metrics.recent_predictions}</Typography>
                <Typography>Error Count: {metrics.error_count}</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Health Checks */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Health Checks
            </Typography>
            {health?.checks && Object.entries(health.checks).map(([key, value]) => (
              <Box key={key} sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    backgroundColor: value.status === 'healthy' ? 'success.main' : 'error.main'
                  }}
                />
                <Typography>
                  {key}: {value.status}
                </Typography>
              </Box>
            ))}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;
