import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Paper, Grid, Card, CardContent, 
  LinearProgress, Alert, List, ListItem, ListItemText
} from '@mui/material';
// Recharts components removed as they are currently unused but kept in import for future use if needed (uncomment to use)
// import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function Monitoring() {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMonitoringData();
    const interval = setInterval(fetchMonitoringData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const fetchMonitoringData = async () => {
    try {
      const [healthRes, metricsRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/health`),
        axios.get(`${API_BASE_URL}/metrics`)
      ]);

      setHealth(healthRes.data);
      setMetrics(metricsRes.data);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch monitoring data');
      setLoading(false);
    }
  };

  if (loading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Monitoring
      </Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      <Grid container spacing={3}>
        {/* System Health */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              
              {health && (
                <Box>
                  <Typography color={health.status === 'healthy' ? 'success.main' : 'error.main'}
                    variant="h5" gutterBottom>
                    {health.status.toUpperCase()}
                  </Typography>
                  
                  <Typography color="textSecondary">
                    Uptime: {health.uptime}
                  </Typography>
                  
                  <Typography color="textSecondary">
                    Last Updated: {new Date(health.timestamp).toLocaleString()}
                  </Typography>

                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2">Health Checks:</Typography>
                    <List dense>
                      {Object.entries(health.checks || {}).map(([key, value]) => (
                        <ListItem key={key}>
                          <ListItemText 
                            primary={`${key}: ${value.status}`}
                            sx={{ 
                              color: value.status === 'healthy' ? 'success.main' : 
                                     value.status === 'warning' ? 'warning.main' : 'error.main'
                            }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Metrics
              </Typography>
              
              {metrics && (
                <Box>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="h3" color="primary">
                        {metrics.prediction_count?.toLocaleString() || 0}
                      </Typography>
                      <Typography color="textSecondary">Total Predictions</Typography>
                    </Grid>
                    
                    <Grid item xs={6}>
                      <Typography variant="h3" color={metrics.error_rate > 0.05 ? 'error' : 'success'}>
                        {((metrics.error_rate || 0) * 100).toFixed(2)}%
                      </Typography>
                      <Typography color="textSecondary">Error Rate</Typography>
                    </Grid>
                    
                    <Grid item xs={6}>
                      <Typography variant="h3">
                        {(metrics.average_latency_ms || 0).toFixed(0)}ms
                      </Typography>
                      <Typography color="textSecondary">Avg Latency</Typography>
                    </Grid>
                    
                    <Grid item xs={6}>
                      <Typography variant="h3">
                        {metrics.recent_predictions || 0}
                      </Typography>
                      <Typography color="textSecondary">Recent (Last 1000)</Typography>
                    </Grid>
                  </Grid>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* System Resources */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Resources
            </Typography>
            
            {health?.system && (
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2">CPU Usage</Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={health.system.cpu_percent || 0}
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                  <Typography>{health.system.cpu_percent?.toFixed(1) || 0}%</Typography>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2">Memory Usage</Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={health.system.memory_percent || 0}
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                  <Typography>{health.system.memory_percent?.toFixed(1) || 0}%</Typography>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2">Disk Usage</Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={health.system.disk_usage || 0}
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                  <Typography>{health.system.disk_usage?.toFixed(1) || 0}%</Typography>
                </Grid>
              </Grid>
            )}
          </Paper>
        </Grid>

        {/* Data Quality Alerts */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Data Quality & Drift Detection
            </Typography>
            <Typography color="textSecondary">
              Real-time data drift detection and model performance monitoring coming in future updates.
            </Typography>
            
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2">Planned Features:</Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="Statistical drift detection (KS test, PSI)" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Feature correlation monitoring" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Prediction distribution tracking" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Automated retraining triggers" />
                </ListItem>
              </List>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Monitoring;
