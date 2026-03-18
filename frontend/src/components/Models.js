import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Paper, Button, List, ListItem, ListItemText,
  IconButton, Dialog, DialogTitle, DialogContent, DialogActions,
  Table, TableBody, TableCell, TableHead, TableRow, Chip,
  LinearProgress, Alert
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function Models() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/models`);
      // Normalize: API may return an array OR { models: [...], count: N }
      const data = response.data;
      setModels(Array.isArray(data) ? data : (data.models || []));
    } catch (err) {
      setError('Failed to fetch models');
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploadLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post(`${API_BASE_URL}/models/load`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setSuccess('Model uploaded successfully');
      fetchModels();
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploadLoading(false);
    }
  };

  const handleDelete = async (modelName) => {
    try {
      await axios.delete(`${API_BASE_URL}/models/${modelName}`);
      setSuccess('Model unloaded successfully');
      fetchModels();
    } catch (err) {
      setError('Failed to unload model');
    }
  };

  const viewModelDetails = (model) => {
    setSelectedModel(model);
    setDialogOpen(true);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Model Management
      </Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Loaded Models ({models.length})
          </Typography>
          <Button
            variant="contained"
            component="label"
            startIcon={<UploadFileIcon />}
            disabled={uploadLoading}
          >
            {uploadLoading ? 'Uploading...' : 'Upload Model'}
            <input type="file" accept=".pkl" hidden onChange={handleUpload} />
          </Button>
        </Box>

        {loading ? (
          <LinearProgress />
        ) : models.length === 0 ? (
          <Alert severity="info">No models loaded. Upload a model to get started.</Alert>
        ) : (
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Model Name</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {models.map((model, index) => (
                <TableRow key={index}>
                  <TableCell>{model}</TableCell>
                  <TableCell>
                    <Chip label="Active" color="success" size="small" />
                  </TableCell>
                  <TableCell>
                    <IconButton onClick={() => viewModelDetails(model)} color="primary" sx={{ mr: 1 }}>
                      <UploadFileIcon />
                    </IconButton>
                    <IconButton onClick={() => handleDelete(model)} color="error">
                      <DeleteIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Model Registry
        </Typography>
        <Typography color="textSecondary">
          Version control and model metadata management coming in future updates.
        </Typography>
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2">Features:</Typography>
          <List dense>
            <ListItem>
              <ListItemText primary="Model versioning and lineage tracking" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Performance comparison across versions" />
            </ListItem>
            <ListItem>
              <ListItemText primary="A/B testing configuration" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Automatic model promotion based on metrics" />
            </ListItem>
          </List>
        </Box>
      </Paper>

      {/* Model Details Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Model Details</DialogTitle>
        <DialogContent>
          {selectedModel && (
            <Box>
              <Typography variant="subtitle1">Name: {selectedModel}</Typography>
              <Typography color="textSecondary">
                Detailed metrics and metadata will be displayed here.
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Models;
