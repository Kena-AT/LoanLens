import React, { useState, useCallback } from 'react';
import { 
  Box, Typography, Paper, Button, Grid, TextField, 
  Slider, Alert, CircularProgress, Card, CardContent, 
  Table, TableBody, TableCell, TableHead, TableRow,
  LinearProgress, Chip
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const REQUIRED_FEATURES = [
  'RevolvingUtilizationOfUnsecuredLines', 'age',
  'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
  'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
  'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
  'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
];

function Predict() {
  const [features, setFeatures] = useState({});
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [fileResult, setFileResult] = useState(null);

  const handleFeatureChange = (name, value) => {
    setFeatures(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        features,
        threshold
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', threshold);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict/file`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setFileResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'File prediction failed');
    } finally {
      setLoading(false);
    }
  }, [threshold]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false
  });

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Make Predictions
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* File Upload */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Batch Prediction (CSV Upload)
            </Typography>
            <Box
              {...getRootProps()}
              sx={{
                p: 3,
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'grey.300',
                borderRadius: 2,
                textAlign: 'center',
                cursor: 'pointer',
                '&:hover': { borderColor: 'primary.main' }
              }}
            >
              <input {...getInputProps()} />
              <UploadFileIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
              <Typography>
                {isDragActive ? 'Drop CSV file here' : 'Drag & drop CSV file, or click to select'}
              </Typography>
            </Box>

            {fileResult && (
              <Box sx={{ mt: 2 }}>
                <Alert severity="success">
                  Processed {fileResult.count} records using {fileResult.model_name}
                </Alert>
                <Table size="small" sx={{ mt: 2 }}>
                  <TableHead>
                    <TableRow>
                      <TableCell>Index</TableCell>
                      <TableCell>Prediction</TableCell>
                      <TableCell>Probability</TableCell>
                      <TableCell>Risk Level</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {fileResult.predictions.slice(0, 10).map((pred) => (
                      <TableRow key={pred.index}>
                        <TableCell>{pred.index}</TableCell>
                        <TableCell>
                          <Chip 
                            label={pred.prediction === 1 ? 'DEFAULT' : 'NO DEFAULT'}
                            color={pred.prediction === 1 ? 'error' : 'success'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{(pred.probability * 100).toFixed(2)}%</TableCell>
                        <TableCell>
                          {pred.probability > 0.7 ? 'High' : pred.probability > 0.3 ? 'Medium' : 'Low'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Single Prediction */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Single Prediction
            </Typography>
            
            <Grid container spacing={2}>
              {REQUIRED_FEATURES.map((feature) => (
                <Grid item xs={12} sm={6} key={feature}>
                  <TextField
                    fullWidth
                    label={feature.replace(/([A-Z])/g, ' $1').trim()}
                    type="number"
                    value={features[feature] || ''}
                    onChange={(e) => handleFeatureChange(feature, e.target.value)}
                    size="small"
                  />
                </Grid>
              ))}
            </Grid>

            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>
                Classification Threshold: {threshold}
              </Typography>
              <Slider
                value={threshold}
                onChange={(_, value) => setThreshold(value)}
                min={0}
                max={1}
                step={0.01}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' }
                ]}
              />
            </Box>

            <Button
              variant="contained"
              onClick={handlePredict}
              disabled={loading || Object.keys(features).length < REQUIRED_FEATURES.length}
              sx={{ mt: 2 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Predict'}
            </Button>
          </Paper>
        </Grid>

        {/* Result */}
        <Grid item xs={12} md={4}>
          {result && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Prediction Result
                </Typography>
                
                <Box sx={{ textAlign: 'center', py: 2 }}>
                  <Typography variant="h2" color={result.prediction === 1 ? 'error' : 'success'}>
                    {result.prediction === 1 ? 'DEFAULT' : 'NO DEFAULT'}
                  </Typography>
                </Box>

                <LinearProgress 
                  variant="determinate" 
                  value={result.probability * 100}
                  sx={{ height: 10, borderRadius: 5, mb: 2 }}
                />

                <Typography>
                  Probability: {(result.probability * 100).toFixed(2)}%
                </Typography>
                <Typography color="textSecondary">
                  Model: {result.model_name}
                </Typography>
                <Typography color="textSecondary">
                  Threshold: {result.threshold}
                </Typography>

                {result.explanation && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Top Contributing Features:
                    </Typography>
                    {result.explanation.features
                      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                      .slice(0, 5)
                      .map((feat, idx) => (
                        <Typography key={idx} variant="body2" color="textSecondary">
                          {feat.feature}: {feat.contribution > 0 ? '+' : ''}{feat.contribution.toFixed(4)}
                        </Typography>
                      ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}

export default Predict;
