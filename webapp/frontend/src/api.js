// In local dev, VITE_API_BASE_URL is unset and requests go to relative "/api/..." paths,
// which vite.config.js proxies to http://localhost:8000. In production (a separate
// Cloud Run service for the backend), it's baked in at build time via --build-arg, since
// there's no dev-server proxy once this is a static build served by nginx.
export const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
