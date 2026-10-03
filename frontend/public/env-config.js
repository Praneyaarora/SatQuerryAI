// Runtime environment configuration for SatQuery frontend.
// In local development (Vite), this provides the default backend URL.
// In Docker production deployments, docker-entrypoint.sh overwrites this dynamically at container start.
window.__SATQUERY_CONFIG__ = window.__SATQUERY_CONFIG__ || {
  API_BASE_URL: 'http://localhost:8000'
};
