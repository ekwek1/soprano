# Error Handling and Troubleshooting

## Error Handling

### HTTP Status Codes
The API returns standard HTTP status codes:

- **200 OK**: Request successful
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Endpoint not found
- **422 Unprocessable Entity**: Validation error in request body
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: Service temporarily unavailable

### Common Error Scenarios

#### 400 Bad Request
- **Cause**: Invalid characters in text that could lead to path traversal
- **Solution**: Remove special characters like '../', '/', or '\' from input text

#### 422 Unprocessable Entity
- **Cause**: Validation errors in request body
- **Examples**:
  - Empty input text
  - Input text exceeding 1000 characters
  - Invalid parameter types or values
- **Solution**: Ensure input meets validation requirements

#### 503 Service Unavailable
- **Cause**: Circuit breaker triggered due to repeated failures
- **Solution**: Wait for recovery period (30 seconds by default) or restart service

## Circuit Breaker Pattern

The API implements a circuit breaker to prevent cascading failures:

- **Threshold**: 3 consecutive failures
- **Recovery Timeout**: 30 seconds
- **States**: CLOSED (normal operation), OPEN (tripped), HALF_OPEN (testing recovery)

## Retry Mechanism

Transient failures are handled with a retry mechanism:

- **Retries**: 2 attempts
- **Initial Delay**: 1 second
- **Backoff Factor**: 2 (exponential backoff)

## Troubleshooting

### Common Issues and Solutions

#### Issue: "CUDA is not available, falling back to CPU"
- **Description**: GPU not detected or CUDA not properly installed
- **Solution**: 
  1. Verify NVIDIA GPU is installed
  2. Install/update CUDA drivers
  3. Ensure CUDA version is 11.3 or higher

#### Issue: "Connection refused" when accessing API
- **Description**: API server not running
- **Solution**: 
  1. Ensure server is started with `python soprano\server\api.py`
  2. Check that port 8000 is available
  3. Verify firewall settings

#### Issue: "Failed to save audio file"
- **Description**: Permission or disk space issues
- **Solution**: 
  1. Verify write permissions to `audio_output` directory
  2. Check available disk space
  3. Ensure directory path is valid

#### Issue: High memory usage
- **Description**: Model consuming excessive memory
- **Solution**: 
  1. Monitor memory usage during operation
  2. Consider reducing concurrent requests
  3. Close other applications to free memory

### Debugging Tips

#### Enable Verbose Logging
The API uses INFO level logging by default. For more detailed debugging:
1. Modify the logging level in the source code if needed
2. Check the console output for detailed error messages

#### Check Model Loading
If experiencing slow responses on first request:
1. Verify model download completed successfully
2. Check internet connectivity during initial model loading

#### Monitor File System
To monitor file creation:
1. Watch the `audio_output` directory
2. Verify sequential file naming is working correctly
3. Check for any permission issues

### Performance Monitoring

#### API Response Times
- First request after startup may be slower due to model loading
- Subsequent requests should be faster
- Monitor for any degradation over time

#### Resource Utilization
- CPU/GPU usage during processing
- Memory consumption
- Disk I/O for file operations