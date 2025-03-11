# Basic Pitch API Service

A containerized service for audio-to-MIDI conversion using Spotify's Basic Pitch model.

## Overview

This service provides an API for converting audio files to MIDI using Spotify's Basic Pitch model. The service is containerized using Docker to ensure consistency between development and deployment environments.

## Requirements

- Docker
- Docker Compose
- Python 3.10 (for local development outside of Docker)
- Sample audio file for testing (MP3 format)

## Development Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/audio-to-midi-api-service.git
   cd audio-to-midi-api-service
   ```

2. Create and configure environment variables:
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Add a test audio file:
   Place an MP3 file in one of these locations:
   - `./test_audio.mp3`
   - `./test.mp3`
   - `./sample_audio/test.mp3`

4. Build and test the Docker container:
   ```
   ./setup_and_test_container.sh
   ```

This will:
- Check for a test audio file
- Build the Docker container
- Run tests to verify the Basic Pitch model works
- Start the API service
- Check if the service is running

## Production Deployment

For production deployment, use the provided `docker-compose.prod.yml` file:

1. Configure your production environment variables:
   ```
   # Edit .env with your production configuration
   ```

2. Deploy using the deployment script:
   ```
   ./deploy.sh
   ```

3. To follow logs:
   ```
   ./deploy.sh --follow-logs
   ```

## Deployment with GitHub Actions

This project uses GitHub Actions to automatically deploy to Replicate. To set it up:

1. Get your Replicate API token from [https://replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)

2. Add the token as a secret in your GitHub repository:
   - Go to your GitHub repository
   - Navigate to Settings > Secrets and variables > Actions
   - Click "New repository secret"
   - Name: `REPLICATE_API_TOKEN`
   - Value: Your Replicate API token
   - Click "Add secret"

3. Push your code to GitHub. The GitHub Actions workflow will automatically deploy your model to Replicate.

4. You can also manually trigger the workflow from the "Actions" tab in your GitHub repository.

## API Endpoints

- `POST /convert`: Convert an audio file to MIDI
- `GET /job/{job_id}`: Get the status of a conversion job
- `GET /health`: Health check endpoint

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| ALLOW_ANONYMOUS_ACCESS | Allow anonymous access to the API | false |
| ALLOW_INTERNAL_NETWORK | Allow access from internal networks | false |
| SUPABASE_URL | Supabase URL for authentication | - |
| SUPABASE_JWT_SECRET | Supabase JWT secret for authentication | - |
| MAX_AUDIO_SIZE_MB | Maximum audio file size in MB | 10 |
| MAX_CONVERSION_TIME_SECONDS | Maximum conversion time in seconds | 300 |
| CLEANUP_COMPLETED_JOBS_AFTER_SECONDS | Time to keep completed jobs in seconds | 3600 |
| ALLOWED_ORIGINS | Comma-separated list of allowed origins for CORS | - |

## Container Configuration

The Docker container includes:
- Python 3.10
- TensorFlow 2.12.0
- Basic Pitch 0.4.0
- All necessary dependencies

## Troubleshooting

If you encounter issues with the containerized service:

1. Verify that environment variables are set correctly:
   ```
   docker-compose -f docker-compose.prod.yml config
   ```

2. Check container logs:
   ```
   docker-compose -f docker-compose.prod.yml logs
   ```

3. Run the test container to verify that Basic Pitch works:
   ```
   docker-compose run test
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 